import assert from "node:assert/strict"
import { cp, mkdtemp, rm, symlink } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { test } from "node:test"
import { fileURLToPath, pathToFileURL } from "node:url"
import { Media, Message } from "@opencode/ai"
import { fromV2SessionMessages, toV2RequestMessages } from "../src/v2/messages"

test("V2 requests preserve host-owned media across separate SDK instances", async (t) => {
    const sandbox = await mkdtemp(join(tmpdir(), "acp-host-media-test-"))
    try {
        // Use the installed minimum SDK in an independent module tree. A second
        // class identity is essential: a shared import hid the original defect.
        const sdk = dirname(dirname(fileURLToPath(import.meta.resolve("@opencode/ai"))))
        const host = join(sandbox, "host-sdk")
        await cp(join(sdk, "dist"), join(host, "dist"), { recursive: true })
        await cp(join(sdk, "package.json"), join(host, "package.json"))
        await symlink(join(sdk, "node_modules"), join(host, "node_modules"), "dir")
        await symlink(resolve(sdk, "../.."), join(sandbox, "node_modules"), "dir")
        const { Media: HostMedia, Message: HostMessage } = (await import(
            pathToFileURL(join(host, "dist/index.js")).href
        )) as typeof import("@opencode/ai")
        assert.notEqual(HostMedia.Asset, Media.Asset)

        const durable = [
            {
                id: "user",
                type: "user",
                time: { created: 1000 },
                text: "Inspect these attachments.",
                files: [
                    { data: "iVBORw0KGgo=", mime: "image/png", name: "diagram.png" },
                    { data: "JVBERi0xLjc=", mime: "application/pdf", name: "paper.pdf" },
                ],
            },
        ]
        const hostMedia = [
            HostMedia.base64("iVBORw0KGgo=", "image/png"),
            HostMedia.bytes(Buffer.from("JVBERi0xLjc=", "base64"), "application/pdf"),
        ]
        const originals = [
            HostMessage.make({
                id: "user",
                role: "user",
                content: [
                    { type: "text", text: durable[0].text },
                    ...hostMedia.map((media, index) => ({
                        type: "media" as const,
                        media,
                        filename: durable[0].files[index].name,
                        metadata: { attachment: index },
                    })),
                ],
            }),
        ]
        const messages = fromV2SessionMessages(durable, "session", originals)

        await t.test("a plugin-owned asset is not valid in the host SDK", () => {
            const standalone = toV2RequestMessages(messages)
            assert.doesNotThrow(() => Message.make(standalone[0] as Message.Input))
            assert.throws(() => HostMessage.make(standalone[0] as Message.Input))
        })

        await t.test("host normalization preserves native instances and exact bytes", () => {
            const before = JSON.stringify(originals)
            const request = toV2RequestMessages(messages, originals)
            const normalized = request.map((message) => HostMessage.make(message as Message.Input))
            const media = normalized[0].content.filter((part) => part.type === "media")
            assert.equal(media.length, 2)
            for (const [index, part] of media.entries()) {
                assert.equal(part.media, hostMedia[index])
                assert.ok(part.media instanceof HostMedia.Asset)
                assert.ok(!(part.media instanceof Media.Asset))
                assert.equal(part.media.inline()?.base64, durable[0].files[index].data)
                assert.equal(part.media.mediaType, durable[0].files[index].mime)
                assert.equal(part.filename, durable[0].files[index].name)
                assert.deepEqual(part.metadata, { attachment: index })
            }
            assert.equal(JSON.stringify(originals), before)
        })

        await t.test("text edits and media removal do not restore original content", () => {
            const transformed = fromV2SessionMessages(durable, "session")
            transformed[0].parts[0].text = "Filtered text"
            transformed[0].parts.splice(1, 1)
            const request = toV2RequestMessages(transformed, originals)
            const normalized = HostMessage.make(request[0] as Message.Input)
            assert.equal(normalized.content.length, 2)
            assert.deepEqual(normalized.content[0], { type: "text", text: "Filtered text" })
            assert.equal(normalized.content[1].type, "media")
            assert.equal(normalized.content[1].media, hostMedia[1])
            assert.deepEqual(toV2RequestMessages([], originals), [])
        })

        await t.test("media matches source bytes and message identity, not array position", () => {
            const other = HostMessage.make({
                id: "other",
                role: "user",
                content: [{ type: "media", media: HostMedia.base64("b3RoZXI=", "image/png") }],
            })
            const request = toV2RequestMessages(messages, [other, ...originals])
            const normalized = HostMessage.make(request[0] as Message.Input)
            assert.equal(normalized.content[1].media, hostMedia[0])
            assert.throws(
                () => toV2RequestMessages(messages, [other]),
                /ACP: missing host media for message user/,
            )
            const changed = fromV2SessionMessages(durable, "session")
            changed[0].parts[1].url = "data:image/png;base64,ZGlmZmVyZW50"
            assert.throws(
                () => toV2RequestMessages(changed, originals),
                /ACP: missing host media for message user/,
            )
        })

        await t.test("host attachment lowering keeps deduplication and decoded text", () => {
            const files = [
                durable[0].files[0],
                durable[0].files[0],
                { data: "dGV4dCBmaWxl", mime: "text/plain", name: "note.txt" },
                { data: "ZGlyZWN0b3J5", mime: "application/x-directory", name: "project" },
            ]
            const loweredText = [
                "Attached file: note.txt\n\ntext file",
                "Attached directory: project\n\ndirectory",
            ]
            const original = HostMessage.make({
                id: "user",
                role: "user",
                content: [
                    { type: "text", text: durable[0].text },
                    { type: "media", media: hostMedia[0], filename: files[0].name },
                    ...loweredText.map((text) => ({ type: "text" as const, text })),
                ],
            })
            const transformed = fromV2SessionMessages([{ ...durable[0], files }], "session", [
                original,
            ])
            const request = toV2RequestMessages(transformed, [original])
            const normalized = HostMessage.make(request[0] as Message.Input)
            assert.equal(normalized.content.filter((part) => part.type === "media").length, 1)
            assert.deepEqual(
                normalized.content.filter((part) => part.type === "text").map((part) => part.text),
                [durable[0].text, ...loweredText],
            )
            assert.equal(normalized.content[1].media, hostMedia[0])
            transformed[0].parts.pop()
            const edited = HostMessage.make(
                toV2RequestMessages(transformed, [original])[0] as Message.Input,
            )
            assert.equal(edited.content.length, 3)
        })
    } finally {
        await rm(sandbox, { recursive: true, force: true })
    }
})
