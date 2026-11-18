/**
 * Fork-recovery: reconstruct ACP compression state from message history.
 *
 * OpenCode forks copy a durable prefix with new message IDs. When the source
 * state is available, transfer its exact aliases and eligible blocks.
 *
 * Without source state, replay historical `acp_compress` invocations in
 * chronological order. Legacy refs can be assigned in nonhost transcript
 * order; host-aware calls require known aliases when deliveries are present.
 *
 * The rebuilt state is an approximation: protected-content enrichments that
 * were appended to summaries at original-compress-time are not re-derived
 * (only the raw model summary from the tool input is available). This is
 * acceptable — protected tool outputs survive in visible context anyway, and
 * the primary goal (pruning compressed messages to avoid overflow) is met.
 */
import type { GCConfig, PluginConfig } from "../config"
import type { Logger } from "../logger"
import { assignMessageRefs, parseMessageRef } from "../message-ids"
import { isHostInput } from "../messages/query"
import { syncCompressionBlocks } from "../messages/sync"
import {
    buildSearchContext,
    resolveAnchorMessageId,
    resolveBoundaryIds,
    resolveSelection,
} from "../compress/search"
import { filterProtectedToolMessages } from "../compress/protected-content"
import { resolveRanges } from "../compress/range-utils"
import {
    allocateBlockId,
    allocateRunId,
    applyCompressionState,
    wrapCompressedSummary,
} from "../compress/state"
import { countTokens } from "../context-usage"
import { createHash } from "node:crypto"
import type { PersistedSessionState } from "./persistence"
import { createPruneMessagesState } from "./utils"
import type {
    BoundaryReference,
    CompressRangeToolArgs,
    CompressRangeEntry,
    CompressMessageToolArgs,
    CompressMessageEntry,
    SearchContext,
    SelectionResolution,
} from "../compress/types"
import type { SessionState, WithParts } from "./types"

interface CompressInvocation {
    messageId: string
    callId: string | undefined
    input: unknown
    hostInputRefs: boolean
}

function collectCompressInvocations(messages: WithParts[]): CompressInvocation[] {
    const invocations: CompressInvocation[] = []
    for (const message of messages) {
        const parts = Array.isArray(message.parts) ? message.parts : []
        for (const part of parts) {
            if (part.type !== "tool" || part.tool !== "acp_compress") {
                continue
            }
            if (part.state?.status !== "completed") {
                continue
            }

            const input = part.state?.input
            if (!input || typeof input !== "object") {
                continue
            }

            invocations.push({
                messageId: message.info.id,
                callId: typeof part.callID === "string" ? part.callID : undefined,
                input,
                hostInputRefs: part.state?.metadata?.acpHostInputRefs === true,
            })
        }
    }
    return invocations
}

/** Range-mode entries have `startId`/`endId`; message-mode entries have `messageId`. */
function isRangeInput(input: any): boolean {
    const content = Array.isArray(input?.content) ? input.content : []
    const first = content[0]
    return !!first && typeof first.startId === "string"
}

function extractBoundaryConsumedBlocks(
    startReference: BoundaryReference,
    endReference: BoundaryReference,
): number[] {
    const consumed: number[] = []
    const seen = new Set<number>()
    for (const ref of [startReference, endReference]) {
        if (
            ref.kind === "compressed-block" &&
            ref.blockId !== undefined &&
            !seen.has(ref.blockId)
        ) {
            seen.add(ref.blockId)
            consumed.push(ref.blockId)
        }
    }
    return consumed
}

function dedupeBlockIds(ids: number[]): number[] {
    const seen = new Set<number>()
    const result: number[] = []
    for (const id of ids) {
        if (!Number.isInteger(id) || id <= 0) continue
        if (seen.has(id)) continue
        seen.add(id)
        result.push(id)
    }
    return result
}

function fingerprintMessage(message: WithParts): string {
    const parts = (message.parts ?? []).map((part) => {
        if (part.type === "text") {
            return { type: "text", text: part.text }
        }
        if (part.type === "tool") {
            // Forks can omit historical compress inputs, which is precisely the
            // case parent-state transfer must recover from.
            if (part.tool === "acp_compress") {
                return { type: "tool", tool: part.tool, status: part.state?.status }
            }
            return {
                type: "tool",
                tool: part.tool,
                status: part.state?.status,
                input: part.state?.input,
                output: part.state?.status === "completed" ? part.state.output : undefined,
            }
        }
        return { type: part.type }
    })
    return createHash("sha256")
        .update(JSON.stringify({ role: message.info.role, hostInput: isHostInput(message), parts }))
        .digest("hex")
}

interface ForkIdMap {
    messages: Map<string, string>
    tools: Map<string, string>
}

function mapForkIds(
    state: SessionState,
    parent: PersistedSessionState,
    parentMessages: WithParts[],
    forkMessages: WithParts[],
): ForkIdMap | null {
    const parentAliases = parent.messageIds
    if (!parentAliases) return null

    const messages = new Map<string, string>()
    const tools = new Map<string, string>()
    let hasSavedHostAliases = false

    function recordMatch(parentMessage: WithParts, forkMessage: WithParts): void {
        messages.set(parentMessage.info.id, forkMessage.info.id)
        for (let index = 0; index < parentMessage.parts.length; index++) {
            const parentPart = parentMessage.parts[index]
            const forkPart = forkMessage.parts[index]
            if (
                parentPart?.type === "tool" &&
                typeof parentPart.callID === "string" &&
                forkPart?.type === "tool" &&
                forkPart.tool === parentPart.tool &&
                typeof forkPart.callID === "string"
            ) {
                tools.set(parentPart.callID, forkPart.callID)
            }
        }
    }

    // A fork changes raw IDs but copies the transcript in order. Reuse only
    // exact saved aliases in the verified prefix. Old source state has no
    // aliases for handoffs; a backfilled delivery may have a much newer ref.
    for (const [index, parentMessage] of parentMessages.entries()) {
        const forkMessage = forkMessages[index]
        if (!forkMessage || fingerprintMessage(parentMessage) !== fingerprintMessage(forkMessage)) {
            return null
        }
        const parentId = parentMessage.info.id
        const forkId = forkMessage.info.id
        const ref = parentAliases.byRawId[parentId]
        if (!ref) continue
        if (
            parentAliases.byRef[ref] !== parentId ||
            (state.messageIds.byRawId.has(forkId) &&
                state.messageIds.byRawId.get(forkId) !== ref) ||
            (state.messageIds.byRef.has(ref) && state.messageIds.byRef.get(ref) !== forkId)
        ) {
            return null
        }
        state.messageIds.byRawId.set(forkId, ref)
        state.messageIds.byRef.set(ref, forkId)
        hasSavedHostAliases ||= isHostInput(parentMessage)
        recordMatch(parentMessage, forkMessage)
    }

    // Preserve the legacy ref epoch: new hosts come after issued source
    // aliases, even if a recovery pass had previously advanced nextRef.
    if (hasSavedHostAliases) {
        state.messageIds.nextRef = Math.max(state.messageIds.nextRef, parentAliases.nextRef)
    } else {
        for (const ref of Object.keys(parentAliases.byRef)) {
            const index = parseMessageRef(ref)
            if (index !== null)
                state.messageIds.nextRef = Math.max(state.messageIds.nextRef, index + 1)
        }
        assignMessageRefs(
            state,
            forkMessages.filter((message) => !isHostInput(message)),
        )
    }
    assignMessageRefs(state, forkMessages)
    state.messageIds.nextRef = Math.max(state.messageIds.nextRef, parentAliases.nextRef)

    return { messages, tools }
}

function translateIds(ids: string[], mapped: Map<string, string>): string[] | null {
    const translated = ids.map((id) => mapped.get(id))
    return translated.some((id) => !id) ? null : (translated as string[])
}

/**
 * Restore source compression blocks for invocations inside the copied prefix.
 * Never import post-fork blocks or a block whose coverage cannot be mapped.
 */
export function restoreForkCompressionState(
    state: SessionState,
    forkMessages: WithParts[],
    parent: PersistedSessionState,
    parentMessages: WithParts[],
    logger: Logger,
): number {
    if (!parent.prune.messages || !parent.messageIds) return 0

    const forkAliases: typeof state.messageIds = {
        byRawId: new Map(state.messageIds.byRawId),
        byRef: new Map(state.messageIds.byRef),
        nextRef: state.messageIds.nextRef,
    }
    const mapped = mapForkIds(
        { ...state, messageIds: forkAliases },
        parent,
        parentMessages,
        forkMessages,
    )
    if (!mapped) return 0

    const sourceMessageIds = new Set(parentMessages.map((message) => message.info.id))
    const parentBlocks = Object.values(parent.prune.messages.blocksById)
    const activeParentIds = new Set(parent.prune.messages.activeBlockIds)
    const translatedBlocks = new Map<number, (typeof parentBlocks)[number]>()
    for (const block of parentBlocks) {
        if (!sourceMessageIds.has(block.compressMessageId)) continue
        const anchorMessageId = mapped.messages.get(block.anchorMessageId)
        const compressMessageId = mapped.messages.get(block.compressMessageId)
        const directMessageIds = translateIds(block.directMessageIds, mapped.messages)
        const effectiveMessageIds = translateIds(block.effectiveMessageIds, mapped.messages)
        const directToolIds = translateIds(block.directToolIds, mapped.tools)
        const effectiveToolIds = translateIds(block.effectiveToolIds, mapped.tools)
        const compressCallId = block.compressCallId
            ? mapped.tools.get(block.compressCallId)
            : undefined
        if (
            !anchorMessageId ||
            !compressMessageId ||
            !directMessageIds ||
            !effectiveMessageIds ||
            !directToolIds ||
            !effectiveToolIds ||
            (block.compressCallId && !compressCallId)
        ) {
            if (activeParentIds.has(block.blockId)) return 0
            continue
        }
        translatedBlocks.set(block.blockId, {
            ...block,
            anchorMessageId,
            compressMessageId,
            compressCallId,
            directMessageIds,
            directToolIds,
            effectiveMessageIds,
            effectiveToolIds,
        })
    }

    if (translatedBlocks.size === 0) return 0

    const copiedBlockIds = new Set(translatedBlocks.keys())
    for (const block of translatedBlocks.values()) {
        if (
            [...block.consumedBlockIds, ...block.includedBlockIds].some(
                (blockId) => !copiedBlockIds.has(blockId),
            )
        ) {
            return 0
        }
        block.parentBlockIds = (block.parentBlockIds ?? []).filter((blockId) =>
            copiedBlockIds.has(blockId),
        )
        if (
            block.deactivatedByBlockId !== undefined &&
            !copiedBlockIds.has(block.deactivatedByBlockId)
        ) {
            block.deactivatedByBlockId = undefined
        }
    }

    const messagesState = createPruneMessagesState()
    for (const block of translatedBlocks.values()) {
        messagesState.blocksById.set(block.blockId, block)
    }
    for (const [parentMessageId, entry] of Object.entries(parent.prune.messages.byMessageId)) {
        const forkMessageId = mapped.messages.get(parentMessageId)
        if (!forkMessageId) continue
        const allBlockIds = entry.allBlockIds.filter((blockId) => copiedBlockIds.has(blockId))
        if (allBlockIds.length === 0) continue
        messagesState.byMessageId.set(forkMessageId, {
            tokenCount: entry.tokenCount,
            allBlockIds,
            activeBlockIds: [],
        })
    }
    messagesState.nextBlockId = Math.max(
        parent.prune.messages.nextBlockId,
        Math.max(...copiedBlockIds) + 1,
    )
    messagesState.nextRunId = parent.prune.messages.nextRunId
    // The source may have consumed a copied block in a later, uncopied call.
    syncCompressionBlocks(
        { ...state, prune: { ...state.prune, messages: messagesState } },
        logger,
        forkMessages,
    )
    state.messageIds = forkAliases
    state.prune.messages = messagesState

    logger.info("fork: restored compression state from parent", {
        blocks: copiedBlockIds.size,
        activeBlocks: messagesState.activeBlockIds.size,
    })
    return copiedBlockIds.size
}

/**
 * Replay a single range-mode compress invocation.
 * Mirrors `createCompressRangeTool.execute` (src/compress/range.ts):
 *   resolveRanges → filterProtectedToolMessages → per-entry block allocation.
 *
 * Returns the number of blocks created.
 */
function rebuildRangeInvocation(
    state: SessionState,
    input: CompressRangeToolArgs,
    searchContext: SearchContext,
    invocation: CompressInvocation,
    protectedTools: string[],
    protectedFilePatterns: string[],
    gcConfig: GCConfig | undefined,
    logger: Logger,
): number {
    // Resolve ALL entries against the pre-invocation state (mirrors range.ts:
    // resolveRanges runs before any block from this call is created).
    const plans = resolveRanges(input, searchContext, state)

    const runId = allocateRunId(state)
    let created = 0

    for (const plan of plans) {
        // [Bug 39] Hard-exclude protected tool messages so they survive in
        // visible context instead of being pruned.
        const filteredSelection = filterProtectedToolMessages(
            plan.selection,
            searchContext,
            protectedTools,
            protectedFilePatterns,
        )
        if (filteredSelection.messageIds.length === 0) {
            continue
        }

        // Auto-detect consumed blocks: requiredBlockIds (active blocks whose
        // anchor falls in range) + boundary blocks (when start/end is a bN ref).
        const boundaryConsumed = extractBoundaryConsumedBlocks(
            filteredSelection.startReference,
            filteredSelection.endReference,
        )
        const consumedBlockIds = dedupeBlockIds([
            ...filteredSelection.requiredBlockIds,
            ...boundaryConsumed,
        ])

        const blockId = allocateBlockId(state)
        const storedSummary = wrapCompressedSummary(blockId, plan.entry.summary)
        const summaryTokens = countTokens(storedSummary)

        applyCompressionState(
            state,
            {
                topic: plan.entry.topic ?? input.topic ?? "",
                batchTopic: typeof input.topic === "string" ? input.topic : undefined,
                startId: plan.entry.startId,
                endId: plan.entry.endId,
                mode: "range",
                runId,
                compressMessageId: invocation.messageId,
                compressCallId: invocation.callId,
                summaryTokens,
            },
            filteredSelection,
            plan.anchorMessageId,
            blockId,
            storedSummary,
            consumedBlockIds,
            gcConfig,
        )
        created++
    }

    return created
}

/** Mirrors `resolveMessage` (src/compress/message-utils.ts) but skips invalid
 *  entries gracefully instead of throwing. */
function resolveMessageEntry(
    entry: CompressMessageEntry,
    searchContext: SearchContext,
    state: SessionState,
): { selection: SelectionResolution; anchorMessageId: string } | null {
    const normalizedRef = entry.messageId.trim()
    if (normalizedRef.toUpperCase() === "BLOCKED") {
        return null
    }

    const ref = normalizedRef.toLowerCase()
    if (!/^m\d{4,5}$/.test(ref)) {
        return null
    }

    const messageId = state.messageIds.byRef.get(ref)
    if (!messageId) {
        return null
    }
    if (!searchContext.rawMessagesById.has(messageId)) {
        return null
    }

    try {
        const { startReference, endReference } = resolveBoundaryIds(searchContext, state, ref, ref)
        const selection = resolveSelection(searchContext, startReference, endReference)
        return {
            selection,
            anchorMessageId: resolveAnchorMessageId(startReference),
        }
    } catch {
        return null
    }
}

/**
 * Replay a single message-mode compress invocation (historical backward-compat).
 *   per-entry resolve → block allocation with consumedBlockIds = [].
 *
 * Returns the number of blocks created.
 */
function rebuildMessageInvocation(
    state: SessionState,
    input: CompressMessageToolArgs,
    searchContext: SearchContext,
    invocation: CompressInvocation,
    gcConfig: GCConfig | undefined,
): number {
    const runId = allocateRunId(state)
    let created = 0

    for (const entry of input.content) {
        const resolved = resolveMessageEntry(entry, searchContext, state)
        if (!resolved) {
            continue
        }

        const blockId = allocateBlockId(state)
        const storedSummary = wrapCompressedSummary(blockId, entry.summary)
        const summaryTokens = countTokens(storedSummary)

        applyCompressionState(
            state,
            {
                topic: entry.topic,
                batchTopic: typeof input.topic === "string" ? input.topic : undefined,
                startId: entry.messageId,
                endId: entry.messageId,
                mode: "message",
                runId,
                compressMessageId: invocation.messageId,
                compressCallId: invocation.callId,
                summaryTokens,
            },
            resolved.selection,
            resolved.anchorMessageId,
            blockId,
            storedSummary,
            [],
            gcConfig,
        )
        created++
    }

    return created
}

/**
 * Reconstruct compression state by replaying historical `compress` tool
 * invocations from message history. Called when no persisted state exists
 * (fork scenario).
 *
 * @returns number of compression blocks reconstructed.
 */
export function rebuildCompressionState(
    state: SessionState,
    messages: WithParts[],
    config: PluginConfig,
    logger: Logger,
): number {
    const invocations = collectCompressInvocations(messages)
    if (invocations.length === 0) {
        assignMessageRefs(state, messages)
        return 0
    }

    // Capture reliable aliases before allocating any locally. Legacy tool
    // inputs used a transcript without host inputs; reserve that epoch first,
    // then append host aliases without shifting historical references.
    const existingRefs = new Map(state.messageIds.byRef)
    const hasHostInputs = messages.some(isHostInput)
    const legacyMessages = messages.filter((message) => !isHostInput(message))
    assignMessageRefs(state, legacyMessages)
    assignMessageRefs(state, messages)

    const protectedTools = config.compress.protectedTools
    const protectedFilePatterns = config.protectedFilePatterns
    const gcConfig = config.gc

    let rebuilt = 0

    for (const invocation of invocations) {
        if (invocation.hostInputRefs && hasHostInputs) {
            const content = (invocation.input as { content?: unknown }).content
            const hasReliableRefs =
                existingRefs.size > 0 &&
                Array.isArray(content) &&
                content.every((entry) => {
                    const refs = isRangeInput(invocation.input)
                        ? [entry?.startId, entry?.endId]
                        : [entry?.messageId]
                    return refs.every(
                        (ref) =>
                            typeof ref === "string" &&
                            (parseMessageRef(ref) === null ||
                                existingRefs.has(ref.trim().toLowerCase())),
                    )
                })
            if (!hasReliableRefs) {
                logger.warn("rebuild: host-input aliases unavailable, stopping history replay")
                break
            }
        }
        // Old invocations must not accidentally prune a host input that was
        // absent from their original ref/selection view.
        const searchContext = buildSearchContext(
            state,
            invocation.hostInputRefs ? messages : legacyMessages,
        )

        try {
            if (isRangeInput(invocation.input)) {
                rebuilt += rebuildRangeInvocation(
                    state,
                    invocation.input as CompressRangeToolArgs,
                    searchContext,
                    invocation,
                    protectedTools,
                    protectedFilePatterns,
                    gcConfig,
                    logger,
                )
            } else {
                rebuilt += rebuildMessageInvocation(
                    state,
                    invocation.input as CompressMessageToolArgs,
                    searchContext,
                    invocation,
                    gcConfig,
                )
            }
        } catch (err: any) {
            logger.warn("rebuild: failed to replay compress invocation, skipping", {
                error: err instanceof Error ? err.message : String(err),
            })
        }
    }

    if (rebuilt > 0) {
        logger.info(`rebuild: reconstructed ${rebuilt} compression block(s) from history`)
    }

    return rebuilt
}
