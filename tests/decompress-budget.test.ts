import assert from "node:assert/strict"
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { test } from "node:test"
import { getConfig } from "../src/config"
import { createDecompressTool } from "../src/compress/decompress"
import { projectDecompression } from "../src/compress/decompress-budget"
import { hideConsumedCompressCalls } from "../src/compress/hide-consumed"
import { resolveCompressionTarget } from "../src/commands/compression-targets"
import { Logger } from "../src/logger"
import { assignMessageRefs } from "../src/message-ids"
import { prune } from "../src/messages/prune"
import { syncCompressionBlocks } from "../src/messages/sync"
import { PromptStore } from "../src/prompts/store"
import { createSessionState, SessionStateRegistry, type CompressionBlock } from "../src/state"
import { countAllMessageTokens } from "../src/context-usage"
import { createChatMessageTransformHandler } from "../src/v2/hooks"
import { fromV2SessionMessages } from "../src/v2/messages"

interface BlockSpec {
    ids: string[]
    summary?: string
    consumes?: number[]
    runId?: number
    mode?: "range" | "message"
}

function fixture(
    sandbox: string,
    texts = ["raw source ".repeat(1000)],
    specs: BlockSpec[] = [{ ids: ["raw-0"] }],
) {
    const logger = new Logger(false, "silent")
    const config = getConfig({ directory: sandbox })
    const state = createSessionState()
    state.sessionId = "session"
    state.storageDir = sandbox
    state.modelContextLimit = 2000
    state.systemPromptTokens = 25
    config.compress.completionReserveTokens = 100
    const durable = [
        { id: "user", type: "user", text: "Original task" },
        ...texts.map((text, i) => ({
            id: `raw-${i}`,
            type: "assistant",
            content: [{ type: "text", text }],
        })),
        ...specs.map((spec, i) => ({
            id: `summary-${i + 1}`,
            type: "assistant",
            content: [
                {
                    type: "tool",
                    id: `compress-${i + 1}`,
                    name: "acp_compress",
                    state: {
                        status: "completed",
                        input: {
                            content: [
                                {
                                    startId: `start-${i + 1}`,
                                    endId: `end-${i + 1}`,
                                    summary: spec.summary ?? "Short summary",
                                },
                            ],
                        },
                        content: [{ type: "text", text: "Compressed" }],
                    },
                },
            ],
        })),
        { id: "latest", type: "user", text: "Continue working" },
    ]
    const history = durable.map((message) => ({ ...message, time: { created: 1000 } }))
    const messages = fromV2SessionMessages(history, "session")
    const byId = new Map(messages.map((message) => [message.info.id, message]))
    for (let i = 0; i < specs.length; i++) {
        const spec = specs[i]
        const blockId = i + 1
        const compressedTokens = spec.ids.reduce(
            (total, id) => total + countAllMessageTokens(byId.get(id)!),
            0,
        )
        const block: CompressionBlock = {
            blockId,
            runId: spec.runId ?? blockId,
            mode: spec.mode ?? "range",
            tier: spec.consumes?.length ? 2 : 1,
            active: false,
            deactivatedByUser: false,
            compressedTokens,
            summaryTokens: 5,
            durationMs: 0,
            topic: `Block ${blockId}`,
            startId: `start-${blockId}`,
            endId: `end-${blockId}`,
            anchorMessageId: `summary-${blockId}`,
            compressMessageId: `summary-${blockId}`,
            compressCallId: `compress-${blockId}`,
            includedBlockIds: spec.consumes ?? [],
            consumedBlockIds: spec.consumes ?? [],
            parentBlockIds: [],
            directMessageIds: spec.ids,
            effectiveMessageIds: spec.ids,
            directToolIds: [],
            effectiveToolIds: [],
            createdAt: blockId,
            summary: spec.summary ?? "Short summary",
            survivedCount: 0,
        }
        state.prune.messages.blocksById.set(blockId, block)
        for (const childId of block.consumedBlockIds) {
            state.prune.messages.blocksById.get(childId)!.parentBlockIds.push(blockId)
        }
        for (const id of spec.ids) {
            const entry = state.prune.messages.byMessageId.get(id) ?? {
                tokenCount: countAllMessageTokens(byId.get(id)!),
                allBlockIds: [],
                activeBlockIds: [],
            }
            entry.allBlockIds.push(blockId)
            state.prune.messages.byMessageId.set(id, entry)
        }
    }
    state.prune.messages.nextBlockId = specs.length + 1
    state.prune.messages.nextRunId = specs.length + 1
    syncCompressionBlocks(state, logger, messages)
    assignMessageRefs(state, messages)
    state.stats.totalPruneTokens = 100000
    const tool = createDecompressTool({
        client: { session: { context: async () => history } },
        registry: { get: () => state } as unknown as SessionStateRegistry,
        logger,
        config,
        prompts: new PromptStore(logger, sandbox),
        assertPermission: async () => {},
    })
    const execute = (args: Record<string, unknown>) =>
        tool.execute(args, { sessionID: "session", messageID: "decompress", id: "decompress-call" })
    return { state, config, logger, durable: history, messages, execute }
}

test("decompression preflights projected context without changing live coverage", async (t) => {
    const sandbox = await mkdtemp(join(tmpdir(), "acp-decompress-test-"))
    const keys = ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "OPENCODE_CONFIG_DIR"] as const
    const previous = keys.map((key) => process.env[key])
    for (const key of keys) process.env[key] = sandbox
    try {
        await t.test(
            "full restoration is rejected atomically and ignores stale saved counts",
            async () => {
                const f = fixture(sandbox, undefined, [
                    { ids: ["raw-0"] },
                    { ids: ["raw-0"], consumes: [1] },
                ])
                f.state.prune.messages.byMessageId.get("raw-0")!.tokenCount = 0
                await writeFile(join(sandbox, "session.json"), "unchanged persisted state")
                const before = structuredClone(f.state)
                const result = await f.execute({ blockId: "b2", full: true })
                assert.match(result.content, /Cannot decompress b2.*exceeding the 1900-token/s)
                assert.match(result.content, /stay compressed.*toFile.*Omit full:true/s)
                assert.deepEqual(f.state, before)
                assert.equal(
                    await readFile(join(sandbox, "session.json"), "utf8"),
                    "unchanged persisted state",
                )
                assert.deepEqual(f.messages, fromV2SessionMessages(f.durable, "session"))
            },
        )

        await t.test("one-tier restoration remains safe and reports visible usage", async () => {
            const f = fixture(sandbox, undefined, [
                { ids: ["raw-0"] },
                { ids: ["raw-0"], consumes: [1] },
            ])
            const result = await f.execute({ blockId: "b2" })
            assert.match(result.content, /Decompressed block b2\. Restored 0 message/)
            assert.match(result.content, /Also restored nested block\(s\): b1/)
            const usage = result.content.match(/Context usage: (\d+)% → (\d+)% \(estimated\)/)
            assert.ok(usage)
            assert.ok(Number(usage[1]) < 100 && Number(usage[2]) < 100)
            assert.deepEqual([...f.state.prune.messages.activeBlockIds], [1])
            const persisted = JSON.parse(await readFile(join(sandbox, "session.json"), "utf8"))
            assert.equal(persisted.prune.messages.blocksById["2"].deactivatedByUser, true)
        })

        await t.test(
            "one-tier summary growth is guarded even with zero restored messages",
            async () => {
                const f = fixture(sandbox, undefined, [
                    { ids: ["raw-0"], summary: "large child summary ".repeat(1000) },
                    { ids: ["raw-0"], consumes: [1] },
                ])
                const before = structuredClone(f.state)
                const result = await f.execute({ blockId: "b2" })
                assert.match(result.content, /Cannot decompress b2.*input budget/s)
                assert.deepEqual(f.state, before)
            },
        )

        await t.test("range restoration budgets all overlapping blocks together", async () => {
            const f = fixture(
                sandbox,
                ["left ".repeat(700), "right ".repeat(700)],
                [{ ids: ["raw-0"] }, { ids: ["raw-1"] }],
            )
            f.state.modelContextLimit = 1200
            const before = structuredClone(f.state)
            const result = await f.execute({
                startId: f.state.messageIds.byRawId.get("raw-0"),
                endId: f.state.messageIds.byRawId.get("raw-1"),
            })
            assert.match(result.content, /Cannot decompress b1, b2/)
            assert.deepEqual(f.state, before)
        })

        await t.test("message-mode siblings are budgeted as one grouped restoration", async () => {
            const f = fixture(
                sandbox,
                ["left ".repeat(700), "right ".repeat(700)],
                [
                    { ids: ["raw-0"], mode: "message", runId: 1 },
                    { ids: ["raw-1"], mode: "message", runId: 1 },
                ],
            )
            f.state.modelContextLimit = 1200
            const before = structuredClone(f.state)
            const result = await f.execute({ blockId: "b1" })
            assert.match(result.content, /Cannot decompress b1/)
            assert.deepEqual(f.state, before)
        })

        await t.test("over-budget reducing and no-op restorations are still allowed", async () => {
            const f = fixture(
                sandbox,
                ["Small original"],
                [{ ids: ["raw-0"], summary: "huge summary ".repeat(1000) }],
            )
            f.state.systemPromptTokens = 5000
            const reduced = await f.execute({ blockId: "b1" })
            assert.match(reduced.content, /Decompressed block b1/)
            const noOp = await f.execute({ blockId: "b1", full: true })
            assert.match(noOp.content, /Decompressed block b1\. Restored 0 message/)
            assert.deepEqual([...f.state.prune.messages.activeBlockIds], [])
        })

        await t.test(
            "exports bypass the context guard but keep state and the path whitelist",
            async () => {
                const f = fixture(sandbox)
                const before = structuredClone(f.state)
                const target = join(sandbox, "restored.txt")
                const result = await f.execute({ blockId: "b1", full: true, toFile: target })
                assert.match(result.content, /stay compressed — context unchanged/)
                assert.ok((await readFile(target, "utf8")).includes("raw source"))
                assert.deepEqual(f.state, before)
                const denied = await f.execute({
                    blockId: "b1",
                    toFile: "/srv/tmp/disallowed-decompress-test.txt",
                })
                assert.match(denied.content, /toFile path must be under/)
                assert.deepEqual(f.state, before)
            },
        )

        await t.test(
            "missing source detection still fails without losing summary coverage",
            async () => {
                const f = fixture(sandbox)
                f.durable.splice(1, 1)
                const before = structuredClone(f.state)
                const result = await f.execute({ blockId: "b1" })
                assert.match(result.content, /source message\(s\) are no longer present/)
                assert.deepEqual(f.state, before)
            },
        )

        await t.test(
            "projection uses current content, including reasoning, not old provider usage",
            () => {
                const f = fixture(sandbox, ["Small source"])
                f.messages[1].parts.push({
                    type: "reasoning",
                    text: "restored reasoning ".repeat(1000),
                })
                for (const message of f.messages) {
                    if (message.info.role === "assistant") message.info.tokens.input = 1
                }
                const before = structuredClone(f.state)
                const originals = structuredClone(f.messages)
                const projection = projectDecompression(
                    f.state,
                    f.config,
                    f.logger,
                    f.messages,
                    [resolveCompressionTarget(f.state.prune.messages, 1)!],
                    {},
                )
                assert.ok(projection.afterTokens > projection.inputBudget!)
                assert.ok(projection.beforeTokens < projection.inputBudget!)
                assert.deepEqual(f.state, before)
                assert.deepEqual(f.messages, originals)
            },
        )

        await t.test(
            "model limit, fallback, system cost and completion reserve are honored",
            async () => {
                const f = fixture(sandbox, ["Small source"])
                const targets = [resolveCompressionTarget(f.state.prune.messages, 1)!]
                const project = () =>
                    projectDecompression(f.state, f.config, f.logger, f.messages, targets, {})
                f.state.modelContextLimit = undefined
                f.config.compress.contextLimitFallback = 128000
                f.config.compress.completionReserveTokens = undefined
                assert.equal(project().inputBudget, 128000 - 32768)
                assert.equal(project().contextLimit?.source, "fallback")
                f.state.modelContextLimit = 64000
                assert.equal(project().contextLimit?.source, "model")
                f.config.compress.completionReserveTokens = Number.NaN
                assert.equal(project().completionReserve, 32768)
                f.config.compress.completionReserveTokens = 0
                assert.equal(project().inputBudget, 64000)
                f.state.systemPromptTokens = 1234
                const withSystem = project().afterTokens
                f.state.systemPromptTokens = 0
                assert.equal(withSystem - project().afterTokens, 1234)
                f.state.modelContextLimit = undefined
                f.config.compress.contextLimitFallback = 0
                assert.equal(project().inputBudget, undefined)
                assert.match((await f.execute({ blockId: "b1" })).content, /Decompressed block b1/)
            },
        )

        await t.test("a reserve larger than the window does not disable the guard", async () => {
            const f = fixture(sandbox, ["source ".repeat(100)])
            f.config.compress.completionReserveTokens = 3000
            assert.match((await f.execute({ blockId: "b1" })).content, /0-token input budget/)
            assert.deepEqual([...f.state.prune.messages.activeBlockIds], [1])
        })

        await t.test("active model and provider overrides apply to the guard", async () => {
            const f = fixture(sandbox)
            f.state.modelContextLimit = undefined
            f.state.modelProviderID = "provider"
            f.state.modelID = "active-model"
            Object.assign(f.durable.at(-1)!, {
                model: { providerID: "old-provider", id: "old-model" },
            })
            f.config.compress.contextLimitFallback = 100000
            f.config.compress.providers = {
                provider: {
                    contextLimitFallback: 4000,
                    completionReserveTokens: 200,
                    models: {
                        "active-model": {
                            contextLimitFallback: 1500,
                            completionReserveTokens: 100,
                        },
                    },
                },
            }
            const project = () =>
                projectDecompression(
                    f.state,
                    f.config,
                    f.logger,
                    fromV2SessionMessages(f.durable, "session"),
                    [resolveCompressionTarget(f.state.prune.messages, 1)!],
                    {},
                )
            assert.equal(project().inputBudget, 1400)
            const before = structuredClone(f.state)
            assert.match((await f.execute({ blockId: "b1" })).content, /1400-token input budget/)
            assert.deepEqual(f.state, before)
            f.config.compress.providers.provider.models = {}
            assert.equal(project().inputBudget, 3800)
            f.state.modelProviderID = undefined
            f.state.modelID = undefined
            Object.assign(f.durable.at(-1)!, {
                model: { providerID: "provider", id: "active-model" },
            })
            assert.equal(project().inputBudget, 3800)
        })

        await t.test(
            "catalog-hydrated model switches change the decompression budget",
            async () => {
                const f = fixture(sandbox)
                const registry = new SessionStateRegistry(f.logger, sandbox)
                registry.getOrCreate = async () => f.state
                const client = {
                    model: {
                        list: async () => ({
                            data: [
                                { providerID: "provider", id: "large", limit: { context: 10000 } },
                                { providerID: "provider", id: "small", limit: { context: 1500 } },
                            ],
                        }),
                    },
                }
                const handler = createChatMessageTransformHandler(
                    client,
                    registry,
                    f.logger,
                    f.config,
                    new PromptStore(f.logger, sandbox),
                    { global: undefined, agents: {} },
                )
                f.config.compress.images.enabled = false
                f.config.compress.maxContextLimit = Number.MAX_SAFE_INTEGER
                f.config.compress.minContextLimit = Number.MAX_SAFE_INTEGER
                f.config.compress.iterationNudgeThreshold = Number.MAX_SAFE_INTEGER
                f.config.compress.contextLimitFallback = 3000
                const project = () =>
                    projectDecompression(
                        f.state,
                        f.config,
                        f.logger,
                        f.messages,
                        [resolveCompressionTarget(f.state.prune.messages, 1)!],
                        {},
                    )
                const switchModel = async (modelID: string) => {
                    const output = structuredClone(f.messages)
                    const latest = output.at(-1)!
                    if (latest.info.role !== "user") throw new Error("Expected user fixture")
                    latest.info.model = { providerID: "provider", modelID }
                    await handler({}, { messages: output })
                }

                await switchModel("large")
                assert.equal(project().inputBudget, 9900)
                await switchModel("small")
                assert.equal(project().inputBudget, 1400)
                assert.match(
                    (await f.execute({ blockId: "b1" })).content,
                    /1400-token input budget/,
                )
                assert.deepEqual([...f.state.prune.messages.activeBlockIds], [1])
                await switchModel("unknown")
                assert.equal(project().inputBudget, 2900)
                assert.equal(project().contextLimit?.source, "fallback")
                assert.equal(f.state.modelID, "unknown")
            },
        )

        await t.test("safe projection agrees with the committed visible content", async () => {
            const f = fixture(
                sandbox,
                ["first source", "second source"],
                [
                    { ids: ["raw-0"] },
                    { ids: ["raw-1"] },
                    { ids: ["raw-0", "raw-1"], consumes: [1, 2] },
                ],
            )
            const projection = projectDecompression(
                f.state,
                f.config,
                f.logger,
                f.messages,
                [resolveCompressionTarget(f.state.prune.messages, 3)!],
                { full: true },
            )
            assert.match(
                (await f.execute({ blockId: "b3", full: true })).content,
                /Restored 2 message/,
            )
            const visible = structuredClone(f.messages)
            prune(f.state, f.logger, f.config, visible)
            hideConsumedCompressCalls(f.state, visible)
            assert.equal(
                projection.afterTokens,
                visible.reduce(
                    (total, message) => total + countAllMessageTokens(message),
                    f.state.systemPromptTokens!,
                ),
            )
        })
        assert.ok((await readdir(sandbox)).includes("session.json"))
    } finally {
        keys.forEach((key, i) => {
            if (previous[i] === undefined) delete process.env[key]
            else process.env[key] = previous[i]
        })
        await rm(sandbox, { recursive: true, force: true })
    }
})
