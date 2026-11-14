import assert from "node:assert/strict"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { test } from "node:test"
import { Media, ToolResultValue } from "@opencode/ai"
import { archiveSessionImages } from "../src/compress/image-archive"
import { Logger } from "../src/logger"
import { createSessionState } from "../src/state"
import { countAllMessageTokens } from "../src/context-usage"
import { fromV2SessionMessages, toV2RequestMessages } from "../src/v2/messages"

const image = {
    type: "file",
    uri: "data:image/png;base64,iVBORw0KGgo=",
    mime: "image/png",
    name: "page-3.png",
}

function convertTool(content: unknown, status = "completed") {
    return fromV2SessionMessages(
        [
            {
                id: "assistant",
                type: "assistant",
                content: [
                    {
                        type: "tool",
                        id: "read-page",
                        name: "read",
                        state: { status, input: { path: "page-3.png" }, content },
                    },
                ],
            },
        ],
        "session",
    )
}

function toolResults(messages: ReturnType<typeof fromV2SessionMessages>) {
    const request = toV2RequestMessages(messages) as Array<{
        role: string
        content: Array<{ type: string; id?: string; result?: ToolResultValue }>
    }>
    return {
        request,
        results: request.flatMap((message) =>
            message.content.filter((part) => part.type === "tool-result"),
        ),
    }
}

test("V2 tool images stay typed and retain text/file ordering and tool pairing", () => {
    const content = [
        { type: "text", text: "First page" },
        image,
        { type: "text", text: "Second page" },
        { ...image, uri: "https://example.test/page.jpg", mime: "image/jpeg", name: "page-4.jpg" },
    ]
    const messages = convertTool(content)
    const { request, results } = toolResults(messages)
    assert.deepEqual(
        request.map((message) => message.role),
        ["assistant", "tool"],
    )
    assert.equal(request[0].content[0].id, "read-page")
    assert.equal(results[0].id, "read-page")
    assert.deepEqual(results[0].result, { type: "content", value: content })
    assert.ok(ToolResultValue.is(results[0].result))
    assert.equal(messages[0].parts[0].state.output, "First page\nSecond page")
    assert.ok(!messages[0].parts[0].state.output.includes("base64"))
    assert.deepEqual(content[1], image)
})

test("V2 media never enters text tokenization, even for large inline payloads", () => {
    const large = { ...image, uri: `data:image/png;base64,${"aB7/".repeat(100000)}` }
    const small = convertTool([{ type: "text", text: "Image read successfully" }, image])
    const big = convertTool([{ type: "text", text: "Image read successfully" }, large])
    assert.equal(countAllMessageTokens(big[0]), countAllMessageTokens(small[0]))
    assert.deepEqual(toolResults(big).results[0].result, {
        type: "content",
        value: [{ type: "text", text: "Image read successfully" }, large],
    })
})

test("text transforms do not resurrect original mixed output or discard its media", () => {
    const messages = convertTool([
        image,
        { type: "text", text: "Old verbose output" },
        { type: "text", text: "More old output" },
    ])
    messages[0].parts[0].state.output = "[truncated]"
    assert.deepEqual(toolResults(messages).results[0].result, {
        type: "content",
        value: [image, { type: "text", text: "[truncated]" }],
    })
    messages[0].parts[0].state.output = ""
    assert.deepEqual(toolResults(messages).results[0].result, { type: "content", value: [image] })
    messages[0].parts = []
    assert.deepEqual(toV2RequestMessages(messages), [])
})

test("image-only and typed content results remain media; text/JSON/errors stay compatible", () => {
    assert.deepEqual(toolResults(convertTool([image])).results[0].result, {
        type: "content",
        value: [image],
    })
    const messages = fromV2SessionMessages(
        [
            {
                id: "assistant",
                type: "assistant",
                content: [
                    {
                        type: "tool-result",
                        id: "typed",
                        name: "read",
                        result: { type: "content", value: [image] },
                    },
                    { type: "tool", id: "text", name: "read", result: "plain text" },
                    { type: "tool", id: "json", name: "read", result: { answer: 42 } },
                    {
                        type: "tool",
                        id: "error",
                        name: "read",
                        state: { status: "error", error: "missing file" },
                    },
                ],
            },
        ],
        "session",
    )
    assert.deepEqual(
        toolResults(messages).results.map((part) => part.result),
        [
            { type: "content", value: [image] },
            { type: "text", value: "plain text" },
            { type: "text", value: '{"answer":42}' },
            { type: "error", value: "missing file" },
        ],
    )
    assert.throws(
        () => convertTool([{ type: "file", uri: image.uri }]),
        /ACP: malformed V2 tool file content/,
    )
})

test("pending calls have no results, and media remains scoped to its originating call", () => {
    const messages = convertTool([image])
    messages[0].parts.push({
        type: "tool",
        callID: "pending",
        tool: "read",
        state: { status: "pending", input: {} },
    })
    const { request, results } = toolResults(messages)
    assert.deepEqual(
        request[0].content.map((part) => part.id),
        ["read-page", "pending"],
    )
    assert.deepEqual(
        results.map((part) => part.id),
        ["read-page"],
    )
})

test("user images and PDFs retain inline bytes, MIME and filename on the model request", () => {
    const messages = fromV2SessionMessages(
        [
            {
                id: "user",
                type: "user",
                text: "Read the diagram and PDF.",
                files: [
                    {
                        data: "iVBORw0KGgo=",
                        mime: "image/png",
                        name: "diagram.png",
                        source: { type: "uri", uri: "file:///original/diagram.png" },
                    },
                    {
                        data: "JVBERi0xLjc=",
                        mime: "application/pdf",
                        name: "paper.pdf",
                        source: { type: "inline" },
                    },
                ],
            },
        ],
        "session",
    )
    const request = toV2RequestMessages(messages) as Array<{
        content: Array<{ type: string; media?: Media.Asset; filename?: string }>
    }>
    const files = request[0].content.filter((part) => part.type === "media")
    assert.deepEqual(
        files.map((part) => part.filename),
        ["diagram.png", "paper.pdf"],
    )
    assert.deepEqual(
        files.map((part) => part.media?.inline()),
        [
            {
                mime: "image/png",
                base64: "iVBORw0KGgo=",
                dataUrl: image.uri,
            },
            {
                mime: "application/pdf",
                base64: "JVBERi0xLjc=",
                dataUrl: "data:application/pdf;base64,JVBERi0xLjc=",
            },
        ],
    )
})

test("existing image archival captures V2 tool media without duplicate writes", async () => {
    const sandbox = await mkdtemp(join(tmpdir(), "acp-v2-media-test-"))
    try {
        const messages = convertTool([{ type: "text", text: "Image read successfully" }, image])
        const state = createSessionState()
        state.sessionId = "session"
        state.storageDir = sandbox
        const config = { enabled: true, minBytes: 0, ttlDays: 30 }
        const logger = new Logger(false, "silent")
        assert.equal(archiveSessionImages(state, messages, config, logger), 1)
        const entries = state.imageArchive.assistant
        assert.equal(entries.length, 1)
        assert.deepEqual(await readFile(entries[0].path), Buffer.from("iVBORw0KGgo=", "base64"))
        assert.equal(archiveSessionImages(state, messages, config, logger), 0)
        messages[0].parts.push({ type: "file", url: image.uri, mime: image.mime })
        assert.equal(archiveSessionImages(state, messages, config, logger), 1)
        assert.match(entries[1].path, /-p1-/)
        assert.equal(archiveSessionImages(state, messages, config, logger), 0)
    } finally {
        await rm(sandbox, { recursive: true, force: true })
    }
})
