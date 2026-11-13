import assert from "node:assert/strict"
import { test } from "node:test"
import type { Context } from "@opencode/plugin/promise/plugin"
import { createEventHandler } from "../src/v2/hooks"
import { Logger } from "../src/logger"
import { SessionStateRegistry } from "../src/state"
import { createV2EventHandler } from "../src/v2/events"

function fixture() {
    const logger = new Logger(false, "silent")
    const registry = new SessionStateRegistry(logger, "/unused")
    const fetches: string[] = []
    const context = new Map<string, unknown[]>()
    const ctx = {
        session: {
            async context({ sessionID }: { sessionID: string }) {
                fetches.push(sessionID)
                return context.get(sessionID) ?? []
            },
        },
    } as unknown as Pick<Context, "session">
    return {
        registry,
        fetches,
        context,
        handle: createV2EventHandler(ctx, createEventHandler(registry, logger)),
    }
}

function event(type: string, created = 100, sessionID = "s", messageID = "m", id = "c") {
    return { type, created, data: { sessionID, assistantMessageID: messageID, id } }
}

test("normal V2 tool lifecycles never fetch history and preserve compression timing", async () => {
    const { handle, registry, fetches } = fixture()
    for (const name of ["read", "shell", "acp_compress"]) {
        const started = event("session.tool.input.started", 10, "s", name)
        await handle({ ...started, data: { ...started.data, name } })
        await handle(event("session.tool.called", 100, "s", name))
        await handle(event("session.tool.called", 150, "s", name))
        await handle(event("session.tool.success", 900, "s", name))
    }
    assert.deepEqual(fetches, [])
    assert.equal(registry.compressionTiming.startsByCallId.size, 0)
    assert.deepEqual(
        [...registry.compressionTiming.pendingByCallId.values()],
        [{ messageId: "acp_compress", callId: "c", durationMs: 800 }],
    )
})

test("names are scoped by session and message; failed calls and idle sessions release entries", async () => {
    const { handle, registry, fetches } = fixture()
    for (const [session, message, name] of [
        ["a", "one", "acp_compress"],
        ["b", "one", "read"],
        ["a", "two", "shell"],
    ]) {
        const start = event("session.tool.input.started", 10, session, message)
        await handle({ ...start, data: { ...start.data, name } })
    }
    await handle(event("session.tool.called", 100, "a", "one"))
    await handle(event("session.tool.called", 100, "b", "one"))
    await handle(event("session.tool.called", 100, "a", "two"))
    assert.equal(registry.compressionTiming.startsByCallId.size, 1)
    await handle(event("session.tool.failed", 200, "a", "one"))
    assert.equal(registry.compressionTiming.startsByCallId.size, 0)
    assert.equal(registry.compressionTiming.pendingByCallId.size, 0)
    assert.deepEqual(fetches, [])
    await handle(event("session.idle", 300, "a"))
    await handle(event("session.tool.called", 400, "a", "two"))
    await handle(event("session.tool.success", 400, "b", "one"))
    assert.deepEqual(fetches, ["a"])
    await handle(event("session.tool.called", 500, "a", "one"))
    assert.deepEqual(fetches, ["a", "a"])
})

test("missing start events recover ordinary and compression names once from the exact message", async () => {
    const { handle, registry, context, fetches } = fixture()
    context.set("s", [
        { id: "other", type: "assistant", content: [{ type: "tool", id: "c", name: "read" }] },
        {
            id: "m",
            type: "assistant",
            content: [{ type: "tool-call", id: "c", name: "acp_compress" }],
        },
        { id: "read", type: "assistant", content: [{ type: "tool", id: "c", name: "read" }] },
    ])
    for (const message of ["m", "read"]) {
        await handle(event("session.tool.called", 100, "s", message))
        await handle(event("session.tool.success", 600, "s", message))
    }
    assert.deepEqual(fetches, ["s", "s"])
    assert.equal(registry.compressionTiming.pendingByCallId.get("m:c")?.durationMs, 500)
})

test("pending-name storage is bounded and deletion clears only the owning session", async () => {
    const { handle, fetches } = fixture()
    for (let i = 0; i < 4100; i++) {
        const start = event("session.tool.input.started", 10, "s", `m${i}`)
        await handle({ ...start, data: { ...start.data, name: "read" } })
    }
    await handle(event("session.tool.called", 100, "s", "m0"))
    await handle(event("session.tool.called", 100, "s", "m4099"))
    assert.deepEqual(fetches, ["s"])
    await handle(event("session.deleted", 200))
    await handle(event("session.tool.called", 300, "s", "m4099"))
    assert.deepEqual(fetches, ["s", "s"])
})
