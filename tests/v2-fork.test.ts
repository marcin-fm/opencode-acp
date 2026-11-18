import assert from "node:assert/strict"
import { test } from "node:test"
import { getConfig } from "../src/config"
import { Logger } from "../src/logger"
import { createSessionState, ensureSessionInitialized } from "../src/state"
import { configureNativeStateStorage, type PersistedSessionState } from "../src/state/persistence"
import { restoreForkCompressionState } from "../src/state/rebuild"
import { fromV2SessionMessages } from "../src/v2/messages"
import { fetchV2ForkSourceMessages } from "../src/v2/session"

const sessionID = "ses_parent"
const prompt = { id: "human", type: "user", time: { created: 1 }, text: "Wait for the evidence." }
const completion = {
    id: "completion",
    type: "synthetic",
    time: { created: 4 },
    text: "Found the required invariant.",
    metadata: { source: "subagent" },
}

function completedCompress(id: string, callID: string, text: string, completed: number) {
    return {
        id,
        type: "assistant",
        time: { created: completed - 1, completed },
        content: [
            {
                type: "tool",
                id: callID,
                name: "acp_compress",
                state: { status: "completed", input: {}, content: [{ type: "text", text }] },
            },
        ],
    }
}

test("V2 fork recovery copies exact aliases only through the host's boundary", async () => {
    const response = {
        id: "response",
        type: "assistant",
        time: { created: 6, completed: 7 },
        content: [{ type: "text", text: "Previous answer" }],
    }
    const compress = completedCompress("compress", "call", "Compressed", 9)
    const rows = [prompt, completion, response, compress]
    const unfinished = { ...response, id: "unfinished", time: { created: 5 } }
    const later = completedCompress("later", "later-call", "Compressed newer history", 10)
    const parentMessages = fromV2SessionMessages(rows, sessionID)
    const block = {
        blockId: 1,
        active: false,
        deactivatedByBlockId: 2,
        anchorMessageId: completion.id,
        compressMessageId: compress.id,
        compressCallId: "call",
        directMessageIds: [completion.id],
        effectiveMessageIds: [completion.id],
        directToolIds: [],
        effectiveToolIds: [],
        consumedBlockIds: [],
        includedBlockIds: [],
        parentBlockIds: [2],
        createdAt: 9,
    }
    const parent = {
        messageIds: {
            byRawId: {
                human: "m00001",
                response: "m00002",
                compress: "m00003",
                completion: "m00004",
                later: "m00005",
            },
            byRef: {
                m00001: "human",
                m00002: "response",
                m00003: "compress",
                m00004: "completion",
                m00005: "later",
            },
            nextRef: 6,
        },
        prune: {
            messages: {
                nextBlockId: 3,
                nextRunId: 3,
                activeBlockIds: [2],
                blocksById: {
                    1: block,
                    2: {
                        ...block,
                        blockId: 2,
                        active: true,
                        deactivatedByBlockId: undefined,
                        compressMessageId: later.id,
                        compressCallId: "later-call",
                        parentBlockIds: [],
                        consumedBlockIds: [1],
                        createdAt: 10,
                    },
                },
                byMessageId: {
                    completion: { allBlockIds: [1, 2], activeBlockIds: [2], tokenCount: 30 },
                },
            },
        },
        nudges: { contextLimitAnchors: [] },
        stats: { pruneTokenCounter: 0, totalPruneTokens: 0 },
    } as unknown as PersistedSessionState
    const forkRows = rows.map((row) => ({ ...row, id: `fork-${row.id}` }))
    const forkID = "ses_fork"
    const forkMessages = fromV2SessionMessages(forkRows, forkID)
    const logger = new Logger(false, "silent")
    const fork = createSessionState()
    const diverged = fromV2SessionMessages(
        [forkRows[0], forkRows[2], forkRows[1], forkRows[3]],
        forkID,
    )
    assert.equal(restoreForkCompressionState(fork, diverged, parent, parentMessages, logger), 0)
    assert.equal(fork.messageIds.byRawId.size, 0, "failed restore must not leak tentative aliases")
    const stored = new Map<string, unknown>([[`sessions/${sessionID}`, parent]])
    configureNativeStateStorage({
        get: async (key) => structuredClone(stored.get(key)),
        set: async (key, value) => {
            stored.set(key, structuredClone(value))
        },
    })
    const config = { ...getConfig({}), storagePath: undefined }
    let sourceReads = 0
    const client = {
        session: {
            get: async ({ sessionID: id }: { sessionID: string }) =>
                id === forkID
                    ? { fork: { sessionID, boundary: { type: "through", messageID: compress.id } } }
                    : { parentID: sessionID },
            context: async ({ sessionID: id }: { sessionID: string }) => {
                assert.equal(id, sessionID)
                sourceReads++
                return [prompt, completion, unfinished, response, compress, later]
            },
        },
    }
    try {
        await ensureSessionInitialized(client, fork, forkID, logger, forkMessages, config)
        assert.equal(fork.isSubAgent, false)
        assert.equal(fork.messageIds.byRawId.get("fork-completion"), "m00004")
        assert.equal(fork.messageIds.byRawId.get("fork-response"), "m00002")
        assert.deepEqual(fork.prune.messages.blocksById.get(1)?.directMessageIds, [
            "fork-completion",
        ])
        assert.deepEqual([...fork.prune.messages.activeBlockIds], [1])
        assert.equal(fork.prune.messages.blocksById.has(2), false)
        assert.equal(fork.prune.messages.nextBlockId, 3)
        assert.ok(stored.has(`sessions/${forkID}`))

        const child = createSessionState()
        await ensureSessionInitialized(client, child, "ses_child", logger, forkMessages, config)
        assert.equal(child.isSubAgent, true)
        assert.equal(child.prune.messages.blocksById.size, 0)
        assert.equal(sourceReads, 1, "subagents do not read or copy a parent's transcript")

        const before = await fetchV2ForkSourceMessages(client, {
            sessionID,
            boundary: { type: "before", messageID: compress.id },
        })
        assert.deepEqual(
            before.map((message) => message.info.id),
            [prompt.id, completion.id, response.id],
        )
        const missing = await fetchV2ForkSourceMessages(client, {
            sessionID,
            boundary: { type: "through", messageID: "compacted-away" },
        })
        assert.deepEqual(missing, [])
    } finally {
        configureNativeStateStorage(undefined)
    }
})
