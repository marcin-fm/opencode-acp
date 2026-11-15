import { Media } from "@opencode/ai"
import type { WithParts } from "../state"

type V2Message = {
    id?: string
    type?: string
    sessionID?: string
    time?: { created?: number }
    text?: string
    description?: string
    metadata?: Record<string, unknown>
    agent?: string
    model?: { providerID?: string; id?: string }
    content?: unknown[]
    files?: Array<{
        data?: string
        mime?: string
        name?: string
        source?: { type?: string; uri?: string }
    }>
}

type V2ToolPart = {
    type?: string
    id?: string
    name?: string
    input?: unknown
    result?: unknown
    state?: {
        status?: string
        input?: unknown
        content?: unknown
        error?: unknown
        metadata?: Record<string, unknown>
    }
}

function createdAt(message: V2Message): number {
    return typeof message.time?.created === "number" ? message.time.created : Date.now()
}

function textFromContent(value: unknown): string {
    if (typeof value === "string") return value
    if (Array.isArray(value)) {
        return value
            .map((item) => {
                if (typeof item === "string") return item
                if (item && typeof item === "object" && "text" in item) {
                    const text = (item as { text?: unknown }).text
                    return typeof text === "string" ? text : ""
                }
                return JSON.stringify(item)
            })
            .filter(Boolean)
            .join("\n")
    }
    if (value === undefined || value === null) return ""
    return typeof value === "object" ? JSON.stringify(value) : String(value)
}

function toolContent(value: unknown): Array<Record<string, unknown>> | undefined {
    if (!Array.isArray(value)) return undefined
    const content = value.map((item) => {
        if (item && typeof item === "object" && item.type === "file") {
            if (typeof item.uri !== "string" || typeof item.mime !== "string") {
                throw new Error("ACP: malformed V2 tool file content")
            }
            return {
                type: "file",
                url: item.uri,
                mime: item.mime,
                ...(typeof item.name === "string" ? { filename: item.name } : {}),
            }
        }
        return { type: "text", text: textFromContent([item]) }
    })
    return content.some((item) => item.type === "file") ? content : undefined
}

function toolContentText(content: Array<Record<string, unknown>>): string {
    return content
        .filter((item) => item.type === "text")
        .map((item) => item.text as string)
        .filter(Boolean)
        .join("\n")
}

function convertToolPart(part: V2ToolPart): Record<string, unknown> {
    const state = part.state
    const status = state?.status ?? (part.result !== undefined ? "completed" : "pending")
    const result = part.result as { type?: string; value?: unknown } | undefined
    const output =
        state?.content ?? state?.error ?? (result?.type === "content" ? result.value : part.result)
    const content = toolContent(output)

    return {
        type: "tool",
        callID: part.id ?? "v2-tool-call",
        tool: part.name ?? "tool",
        state: {
            status,
            input: state?.input ?? part.input,
            output: content ? toolContentText(content) : textFromContent(output),
            ...(content ? { content } : {}),
            ...(state?.metadata?.acpHostInputRefs === true
                ? { metadata: { acpHostInputRefs: true } }
                : {}),
        },
    }
}

function convertParts(message: V2Message, hostUserParts?: unknown[]): unknown[] {
    if (message.type === "user" || message.type === "synthetic") {
        if (hostUserParts !== undefined) return hostUserParts
        const parts: unknown[] = []
        if (typeof message.text === "string") {
            parts.push({ type: "text", text: message.text })
        }
        for (const file of message.files ?? []) {
            if (typeof file.data !== "string" || typeof file.mime !== "string") continue
            // Match OpenCode's attachment lowering: the durable bytes are the
            // payload; source.uri is provenance, not a replacement for them.
            parts.push({
                type: "file",
                url: `data:${file.mime};base64,${file.data}`,
                mime: file.mime,
                ...(typeof file.name === "string" ? { filename: file.name } : {}),
            })
        }
        return parts
    }

    const parts: unknown[] = []
    for (const item of message.content ?? []) {
        if (!item || typeof item !== "object") continue
        const part = item as V2ToolPart & { text?: unknown }
        if (part.type === "text") {
            parts.push({ type: "text", text: typeof part.text === "string" ? part.text : "" })
        } else if (part.type === "reasoning") {
            parts.push({ type: "reasoning", text: typeof part.text === "string" ? part.text : "" })
        } else if (
            part.type === "tool" ||
            part.type === "tool-call" ||
            part.type === "tool-result"
        ) {
            parts.push(convertToolPart(part))
        }
    }
    return parts
}

type V2RequestInput = {
    id?: string
    role?: string
    content?: unknown
    metadata?: Record<string, unknown>
}

/** ACP notifications are UI output, not fresh model input. */
function isAcpNotification(message: V2Message): boolean {
    return (
        message.metadata?.source === "opencode-acp" ||
        (message.metadata?.source === undefined &&
            (message.description === "ACP" ||
                message.description === "ACP update ready" ||
                message.description?.startsWith("ACP:") === true))
    )
}

/** Convert durable messages, restricting host deliveries to the active request when supplied. */
export function fromV2SessionMessages(
    input: unknown,
    sessionID: string,
    requestMessages?: readonly unknown[],
): WithParts[] {
    if (!Array.isArray(input)) return []

    const requestInputs = new Map<string, V2RequestInput>()
    for (const value of requestMessages ?? []) {
        if (!value || typeof value !== "object") continue
        const message = value as V2RequestInput
        if (message.role === "user" && typeof message.id === "string") {
            requestInputs.set(message.id, message)
        }
    }

    const result: WithParts[] = []
    const hostUserParts =
        requestMessages === undefined ? undefined : indexHostUserParts(requestMessages)
    for (const value of input) {
        if (!value || typeof value !== "object") continue
        const message = value as V2Message
        const hostInput = message.type === "synthetic"
        if (!hostInput && message.type !== "user" && message.type !== "assistant") continue
        if (typeof message.id !== "string") continue
        const requestInput = requestInputs.get(message.id)
        if (hostInput && (isAcpNotification(message) || (requestMessages && !requestInput)))
            continue

        const role = hostInput ? "user" : message.type
        const info = {
            id: message.id,
            sessionID,
            role,
            ...(hostInput
                ? { hostInput: { metadata: requestInput?.metadata ?? message.metadata } }
                : {}),
            time: { created: createdAt(message) },
            agent: message.agent ?? "code",
            model: {
                providerID: message.model?.providerID ?? "",
                modelID: message.model?.id ?? "",
            },
            parentID: "",
            modelID: message.model?.id ?? "",
            providerID: message.model?.providerID ?? "",
            mode: "code",
            path: { cwd: "", root: "" },
            cost: 0,
            tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        }

        const projectedContent = hostInput ? requestInput?.content : undefined
        const parts =
            typeof projectedContent === "string"
                ? [{ type: "text", text: projectedContent }]
                : Array.isArray(projectedContent)
                  ? projectedContent.map((part) => structuredClone(part))
                  : convertParts(message, hostUserParts?.get(message.id))
        result.push({
            info: info as WithParts["info"],
            parts: parts as WithParts["parts"],
        })
    }
    return result
}

export async function fetchV2SessionMessages(
    client: unknown,
    sessionID: string,
    requestMessages?: readonly unknown[],
): Promise<WithParts[]> {
    const runtime = client as {
        session?: { context?: (input: { sessionID: string }) => Promise<unknown> }
    }
    if (typeof runtime.session?.context !== "function") {
        return []
    }
    return fromV2SessionMessages(
        await runtime.session.context({ sessionID }),
        sessionID,
        requestMessages,
    )
}

function toolIdentity(part: Record<string, unknown>) {
    return {
        id: typeof part.callID === "string" ? part.callID : "v2-tool-call",
        name: typeof part.tool === "string" ? part.tool : "tool",
    }
}

function toV2ToolCall(part: Record<string, unknown>): Record<string, unknown> {
    const state = (part.state ?? {}) as Record<string, unknown>
    return {
        type: "tool-call",
        ...toolIdentity(part),
        input: state.input ?? {},
    }
}

function toV2ToolResult(part: Record<string, unknown>): Record<string, unknown> {
    const state = (part.state ?? {}) as Record<string, unknown>
    const content = state.content as Array<Record<string, unknown>> | undefined
    let result: Record<string, unknown> = {
        type: state.status === "error" ? "error" : "text",
        value: state.output ?? "",
    }
    if (content && state.status !== "error") {
        // ACP edits the text projection (tags, truncation, etc.). Only replay
        // the original text blocks while that projection is unchanged; never
        // resurrect discarded text alongside the preserved media.
        const unchanged = state.output === toolContentText(content)
        let textInserted = false
        const value: Array<Record<string, unknown>> = []
        for (const item of content) {
            if (item.type === "file") {
                value.push({
                    type: "file",
                    uri: item.url,
                    mime: item.mime,
                    ...(typeof item.filename === "string" ? { name: item.filename } : {}),
                })
            } else if (unchanged) {
                value.push({ type: "text", text: item.text })
            } else if (!textInserted) {
                if (state.output) value.push({ type: "text", text: state.output })
                textInserted = true
            }
        }
        if (!unchanged && !textInserted && state.output) {
            value.push({ type: "text", text: state.output })
        }
        result = { type: "content", value }
    }
    return {
        type: "tool-result",
        ...toolIdentity(part),
        result,
    }
}

function mediaKey(url: string): string {
    if (!url.startsWith("data:")) return url
    const separator = url.indexOf(";base64,")
    if (separator === -1) return url
    return url.slice(0, separator).toLowerCase() + url.slice(separator)
}

function hostMediaUrl(value: unknown): string | undefined {
    if (!value || typeof value !== "object") return undefined
    const asset = value as {
        inline?: () => { dataUrl: string } | undefined
        source?: { type?: string; url?: string }
    }
    return (
        (typeof asset.inline === "function" ? asset.inline()?.dataUrl : undefined) ??
        (asset.source?.type === "url" ? asset.source.url : undefined)
    )
}

function indexHostUserParts(messages: readonly unknown[]) {
    const index = new Map<string, unknown[]>()
    for (const value of messages) {
        if (!value || typeof value !== "object") continue
        const message = value as { id?: unknown; role?: unknown; content?: unknown }
        if (
            typeof message.id !== "string" ||
            message.role !== "user" ||
            !Array.isArray(message.content)
        ) {
            continue
        }
        // OpenCode already lowers text/directory attachments and deduplicates
        // inline image mentions. Transform that view, not a second lowering.
        const parts: unknown[] = []
        for (const value of message.content) {
            if (!value || typeof value !== "object") continue
            const part = value as Record<string, unknown>
            if (part.type === "text" && typeof part.text === "string") {
                parts.push({ type: "text", text: part.text })
            } else if (part.type === "media") {
                const url = hostMediaUrl(part.media)
                if (typeof url !== "string") {
                    throw new Error(`ACP: unsupported host media for message ${message.id}`)
                }
                const asset = part.media as { mediaType?: string }
                parts.push({
                    type: "file",
                    url,
                    mime: asset.mediaType ?? "application/octet-stream",
                    ...(typeof part.filename === "string" ? { filename: part.filename } : {}),
                })
            }
        }
        index.set(message.id, parts)
    }
    return index
}

function indexHostMedia(messages: readonly unknown[]) {
    const index = new Map<string, Map<string, Array<Record<string, unknown>>>>()
    for (const value of messages) {
        if (!value || typeof value !== "object") continue
        const message = value as { id?: unknown; content?: unknown }
        if (typeof message.id !== "string" || !Array.isArray(message.content)) continue
        for (const value of message.content) {
            if (!value || typeof value !== "object") continue
            const part = value as Record<string, unknown>
            if (part.type !== "media") continue
            const url = hostMediaUrl(part.media)
            if (typeof url !== "string") continue
            let byUrl = index.get(message.id)
            if (!byUrl) index.set(message.id, (byUrl = new Map()))
            const key = mediaKey(url)
            const entries = byUrl.get(key) ?? []
            entries.push(part)
            byUrl.set(key, entries)
        }
    }
    return index
}

function toV2Content(
    part: Record<string, unknown>,
    hostMedia?: (part: Record<string, unknown>) => Record<string, unknown>,
): Record<string, unknown> | undefined {
    if (part.type === "text") {
        return { type: "text", text: typeof part.text === "string" ? part.text : "" }
    }
    if (part.type === "reasoning") {
        return { type: "reasoning", text: typeof part.text === "string" ? part.text : "" }
    }
    if (part.type === "tool") {
        const state = (part.state ?? {}) as Record<string, unknown>
        return state.status === "completed" || state.status === "error"
            ? toV2ToolResult(part)
            : toV2ToolCall(part)
    }
    if (part.type === "file" && typeof part.url === "string") {
        if (hostMedia) return hostMedia(part)
        return {
            type: "media",
            media:
                Media.parseDataUrl(part.url) ??
                Media.url(part.url, {
                    mediaType: typeof part.mime === "string" ? part.mime : undefined,
                }),
            ...(typeof part.filename === "string" ? { filename: part.filename } : {}),
        }
    }
    return undefined
}

/**
 * Convert the internal post-transform view back to V2 model messages. At the
 * context-hook boundary, media must belong to the host's SDK, not our copy.
 */
export function toV2RequestMessages(
    messages: WithParts[],
    originalMessages?: readonly unknown[],
): unknown[] {
    const result: unknown[] = []
    const hostMedia = originalMessages === undefined ? undefined : indexHostMedia(originalMessages)

    for (const message of messages) {
        const resolveMedia =
            hostMedia === undefined
                ? undefined
                : (part: Record<string, unknown>) => {
                      const entries = hostMedia
                          .get(message.info.id)
                          ?.get(mediaKey(part.url as string))
                      const index =
                          entries?.findIndex((candidate) => candidate.filename === part.filename) ??
                          -1
                      if (!entries || index === -1) {
                          throw new Error(`ACP: missing host media for message ${message.info.id}`)
                      }
                      // Preserve the native asset and attachment metadata. Only
                      // transformed parts are replayed; never restore removed media.
                      return entries.splice(index, 1)[0]
                  }
        const content: Record<string, unknown>[] = []
        const toolResults: Record<string, unknown>[] = []
        for (const value of message.parts) {
            const part = value as unknown as Record<string, unknown>
            if (message.info.role === "assistant" && part.type === "tool") {
                content.push(toV2ToolCall(part))
                const state = (part.state ?? {}) as Record<string, unknown>
                if (state.status === "completed" || state.status === "error") {
                    toolResults.push(toV2ToolResult(part))
                }
                continue
            }
            const converted = toV2Content(part, resolveMedia)
            if (converted !== undefined) content.push(converted)
        }

        // OpenResponses treats tool output as a separate `tool` message. A
        // durable OpenCode assistant message stores a completed tool part next
        // to the assistant's text/tool-call parts, so split it before handing
        // the request to the model adapter. Keeping tool-result inside an
        // assistant message is rejected by OpenAI Responses-compatible models.
        if (message.info.role === "assistant") {
            const assistantContent = content.filter((part) => part.type !== "tool-result")
            if (assistantContent.length > 0) {
                result.push({ role: "assistant", content: assistantContent })
            }
            for (const part of toolResults) {
                result.push({ role: "tool", content: [part] })
            }
            continue
        }

        if (content.length > 0) {
            result.push({
                role: message.info.role,
                content,
                ...(message.info.role === "user" && message.info.hostInput
                    ? { id: message.info.id, metadata: message.info.hostInput.metadata }
                    : {}),
            })
        }
    }

    return result
}
