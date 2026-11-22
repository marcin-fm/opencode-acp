import assert from "node:assert/strict"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { test } from "node:test"
import { getConfig } from "../src/config"
import { buildSearchContext } from "../src/compress/search"
import { prepareExecutableRangePlans } from "../src/compress/range-utils"
import { createCompressRangeTool } from "../src/compress/range"
import { Logger } from "../src/logger"
import { assignMessageRefs } from "../src/message-ids"
import { prune } from "../src/messages/prune"
import { PromptStore } from "../src/prompts/store"
import { createSessionState, type SessionStateRegistry } from "../src/state"
import { fromV2SessionMessages, toV2RequestMessages } from "../src/v2/messages"

test("range planning reports omissions and honors dangerous only for recent soft protection", async () => {
    const sandbox = await mkdtemp(join(tmpdir(), "acp-protection-test-"))
    const keys = ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "OPENCODE_CONFIG_DIR"] as const
    const previous = keys.map((key) => process.env[key])
    for (const key of keys) process.env[key] = sandbox
    try {
        const logger = new Logger(false, "silent")
        const config = getConfig({ directory: sandbox })
        config.compress.minCompressRange = 0
        config.compress.preserveRecentMessages = 5
        config.compress.preserveRecentTokens = 0
        const messages = fromV2SessionMessages(
            [
                { id: "user", type: "user", text: "Read a PDF" },
                ...Array.from({ length: 9 }, (_, i) => ({
                    id: `assistant-${i + 1}`,
                    type: "assistant",
                    content:
                        i === 7
                            ? [
                                  {
                                      type: "tool",
                                      id: "read-image",
                                      name: "read",
                                      state: {
                                          status: "completed",
                                          input: { path: "page.png" },
                                          content: [
                                              {
                                                  type: "file",
                                                  uri: "data:image/png;base64,iVBORw0KGgo=",
                                                  mime: "image/png",
                                              },
                                          ],
                                      },
                                  },
                              ]
                            : [{ type: "text", text: `Work step ${i + 1}` }],
                })),
            ],
            "session",
        )
        const state = createSessionState()
        assignMessageRefs(state, messages)
        const search = buildSearchContext(state, messages)
        const result = prepareExecutableRangePlans(
            {
                topic: "PDF setup",
                content: [{ startId: "m00002", endId: "m00009", summary: "Prepared PDF pages." }],
            },
            search,
            state,
            config,
            logger,
        )
        assert.deepEqual(result.plans[0].selection.messageIds, [
            "assistant-1",
            "assistant-2",
            "assistant-3",
            "assistant-4",
        ])
        assert.deepEqual(result.protectedMessageIds, [
            "assistant-5",
            "assistant-6",
            "assistant-7",
            "assistant-8",
        ])
        const dangerous = prepareExecutableRangePlans(
            {
                topic: "PDF setup",
                dangerous: true,
                content: [{ startId: "m00001", endId: "m00009", summary: "Prepared PDF pages." }],
            },
            search,
            state,
            config,
            logger,
        )
        assert.equal(dangerous.plans[0].selection.messageIds.length, 8)
        assert.ok(dangerous.plans[0].selection.messageIds.includes("assistant-8"))
        assert.deepEqual(dangerous.protectedMessageIds, ["user"])
        config.compress.protectedTools = ["read"]
        const hardProtected = prepareExecutableRangePlans(
            {
                topic: "PDF setup",
                dangerous: true,
                content: [{ startId: "m00001", endId: "m00009", summary: "Prepared PDF pages." }],
            },
            search,
            state,
            config,
            logger,
        )
        assert.ok(!hardProtected.plans[0].selection.messageIds.includes("assistant-8"))
        assert.deepEqual(hardProtected.protectedMessageIds, ["user", "assistant-8"])
        config.compress.protectedTools = []
        assert.throws(
            () =>
                prepareExecutableRangePlans(
                    {
                        topic: "Recent pages",
                        content: [{ startId: "m00009", endId: "m00010", summary: "Read pages." }],
                    },
                    search,
                    state,
                    config,
                    logger,
                ),
            /recent-message protection.*m00009.*m00010/s,
        )
        config.compress.lastSegmentSoftBlock = false
        const unprotected = prepareExecutableRangePlans(
            {
                topic: "PDF setup",
                content: [{ startId: "m00002", endId: "m00009", summary: "Prepared PDF pages." }],
            },
            search,
            state,
            config,
            logger,
        )
        assert.equal(unprotected.plans[0].selection.messageIds.length, 8)
        assert.deepEqual(unprotected.protectedMessageIds, [])

        // Exercise the public tool and its persisted block membership, not
        // merely the planner: a successful call must describe retained content.
        config.compress.lastSegmentSoftBlock = true
        config.qualityGate.enabled = false
        config.pruneNotification = "off"
        state.sessionId = "session"
        state.storageDir = sandbox
        const durable = [
            { id: "user", type: "user", text: "Read a PDF" },
            ...messages.slice(1).map((message) => ({
                id: message.info.id,
                type: "assistant",
                content: message.parts.map((part) =>
                    part.type === "tool"
                        ? {
                              type: "tool",
                              id: part.callID,
                              name: part.tool,
                              state: {
                                  status: "completed",
                                  input: part.state.input,
                                  content: [
                                      {
                                          type: "file",
                                          uri: "data:image/png;base64,iVBORw0KGgo=",
                                          mime: "image/png",
                                      },
                                  ],
                              },
                          }
                        : part,
                ),
            })),
        ]
        const tool = createCompressRangeTool({
            client: { session: { context: async () => durable } },
            registry: { get: () => state } as unknown as SessionStateRegistry,
            logger,
            config,
            prompts: new PromptStore(logger, sandbox),
            assertPermission: async () => {},
        })
        const ctx = { sessionID: "session", messageID: "compress", id: "compress-call" }
        const partial = await tool.execute(
            {
                topic: "PDF preparation",
                content: [{ startId: "m00002", endId: "m00009", summary: "Prepared the PDF." }],
            },
            ctx,
        )
        assert.match(partial.content, /Compressed 4 messages/)
        assert.match(partial.content, /4 requested message\(s\) were NOT compressed.*m00009/s)
        assert.match(partial.content, /including any images, remains in visible context/)
        const retained = fromV2SessionMessages(durable, "session")
        prune(state, logger, config, retained)
        assert.ok(retained.some((message) => message.info.id === "assistant-8"))
        const complete = await tool.execute(
            {
                topic: "Consumed PDF pages",
                dangerous: true,
                content: [{ startId: "m00006", endId: "m00009", summary: "Read the PDF pages." }],
            },
            ctx,
        )
        assert.match(complete.content, /Compressed 4 messages/)
        assert.ok(!complete.content.includes("NOT compressed"))
        const pruned = fromV2SessionMessages(durable, "session")
        prune(state, logger, config, pruned)
        assert.ok(!pruned.some((message) => message.info.id === "assistant-8"))
        assert.ok(!JSON.stringify(toV2RequestMessages(pruned)).includes("base64"))
    } finally {
        keys.forEach((key, i) => {
            if (previous[i] === undefined) delete process.env[key]
            else process.env[key] = previous[i]
        })
        await rm(sandbox, { recursive: true, force: true })
    }
})
