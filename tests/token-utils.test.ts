import assert from "node:assert/strict"
import { test } from "node:test"
import { countTokens as referenceCountTokens } from "@anthropic-ai/tokenizer"
import { createSessionState } from "../src/state"
import {
    COMPACTED_TOOL_OUTPUT_PLACEHOLDER,
    countAllMessageTokens,
    countTokens,
    countToolTokens,
    estimateTokensBatch,
    getCurrentTokenUsage,
} from "../src/context-usage"
import { fromV2SessionMessages } from "../src/v2/messages"

test("exact counts preserve normalization, special tokens, and batch boundaries", () => {
    const samples = [
        "",
        "An English sentence with punctuation: 12345!",
        "中文和日本語。한국어 العربية",
        "ＡＢＣ ﬁ ① e\u0301 café 👨‍👩‍👧‍👦",
        "<|endoftext|><|fim_prefix|> code <|fim_suffix|>",
        "const value = await readFile(path);\n".repeat(80),
    ]
    for (const text of samples) {
        const expected = text ? referenceCountTokens(text) : 0
        assert.equal(countTokens(text), expected)
        assert.equal(countTokens(text), expected)
    }
    assert.equal(estimateTokensBatch([]), 0)
    assert.equal(estimateTokensBatch(samples), referenceCountTokens(samples.join(" ")))
})

test("content changes, tool states, pruning, and compaction cannot reuse stale counts", () => {
    const messages = fromV2SessionMessages(
        [
            { id: "user", type: "user", text: "Inspect this output.", time: { created: 1 } },
            {
                id: "assistant",
                type: "assistant",
                time: { created: 2 },
                content: [
                    { type: "text", text: "Original answer." },
                    {
                        type: "tool",
                        id: "read-call",
                        name: "read",
                        input: { path: "sample.txt" },
                        result: "Large record 中文. ".repeat(50),
                    },
                ],
            },
        ],
        "session",
    )
    const state = createSessionState()
    const assistant = messages[1]!
    const tool = assistant.parts[1]!
    const original = countAllMessageTokens(assistant)
    assert.equal(
        original,
        referenceCountTokens(
            ["Original answer.", JSON.stringify(tool.state.input), tool.state.output].join(" "),
        ),
    )
    assistant.parts[0].text = "A changed answer with the same message ID."
    tool.state.input = { path: "replacement.txt" }
    tool.state.output = "Short replacement."
    const changed = referenceCountTokens(
        [assistant.parts[0].text, JSON.stringify(tool.state.input), tool.state.output].join(" "),
    )
    assert.notEqual(changed, original)
    assert.equal(countAllMessageTokens(assistant), changed)
    assert.equal(countAllMessageTokens(structuredClone(assistant)), changed)
    assert.equal(
        getCurrentTokenUsage(state, messages),
        countAllMessageTokens(messages[0]!) + changed,
    )
    assert.equal(getCurrentTokenUsage(state, [messages[0]!]), countAllMessageTokens(messages[0]!))

    tool.state.time = { compacted: 3 }
    assert.equal(
        countToolTokens(tool),
        referenceCountTokens(
            [JSON.stringify(tool.state.input), COMPACTED_TOOL_OUTPUT_PLACEHOLDER].join(" "),
        ),
    )
    tool.state.status = "error"
    tool.state.error = "Permission denied."
    assert.equal(
        countToolTokens(tool),
        referenceCountTokens([JSON.stringify(tool.state.input), tool.state.error].join(" ")),
    )
    tool.state.status = "pending"
    assert.equal(countToolTokens(tool), referenceCountTokens(JSON.stringify(tool.state.input)))
})

test("recorded usage and last-compaction behavior remain unchanged", () => {
    const [message] = fromV2SessionMessages(
        [{ id: "assistant", type: "assistant", time: { created: 10 }, content: [] }],
        "session",
    )
    assert.ok(message)
    assert.equal(message.info.role, "assistant")
    if (message.info.role !== "assistant") return
    message.info.tokens = { input: 20, output: 0, reasoning: 3, cache: { read: 40, write: 5 } }
    const state = createSessionState()
    assert.equal(getCurrentTokenUsage(state, [message]), 68)
    state.lastCompaction = 11
    assert.equal(getCurrentTokenUsage(state, [message]), 0)
})
