import assert from "node:assert/strict"
import { createRequire } from "node:module"
import { test } from "node:test"

test("tokenizer reuse and bounded LRU recover from failures without caching estimates", async (t) => {
    const upstream = createRequire(import.meta.url)(
        "@anthropic-ai/tokenizer",
    ) as typeof import("@anthropic-ai/tokenizer")
    const originalGetTokenizer = upstream.getTokenizer
    let constructions = 0
    let encodes = 0
    let releases = 0
    let failConstruction = true
    let failEncoding = false
    t.mock.method(upstream, "getTokenizer", () => {
        constructions++
        if (failConstruction) throw new Error("synthetic constructor failure")
        const tokenizer = originalGetTokenizer()
        const encode = tokenizer.encode.bind(tokenizer)
        const free = tokenizer.free.bind(tokenizer)
        t.mock.method(tokenizer, "encode", (...args: Parameters<typeof encode>) => {
            encodes++
            if (failEncoding) throw new Error("synthetic encoding failure")
            return encode(...args)
        })
        t.mock.method(tokenizer, "free", () => {
            releases++
            free()
        })
        return tokenizer
    })
    const { countTokens } = await import("../src/context-usage")
    const text = "中文".repeat(10)
    const exact = upstream.countTokens(text)
    assert.notEqual(exact, Math.round(text.length / 4))
    assert.equal(countTokens(""), 0)
    assert.equal(constructions, 0, "empty inputs do not initialize WASM")
    assert.equal(countTokens(text), Math.round(text.length / 4))
    failConstruction = false
    assert.equal(countTokens(text), exact, "failed estimates are retried")
    assert.equal(constructions, 2)
    const beforeRepeat = encodes
    assert.equal(countTokens(text), exact)
    assert.equal(encodes, beforeRepeat, "identical text is not re-encoded")

    failEncoding = true
    const retryText = `${text} retry`
    assert.equal(countTokens(retryText), Math.round(retryText.length / 4))
    assert.equal(releases, 1)
    failEncoding = false
    assert.equal(countTokens(retryText), upstream.countTokens(retryText))
    assert.equal(constructions, 3, "a failed encoder is replaced")

    // More than the entry cap must evict old content while a frequently used
    // entry survives. Counts after eviction must still be exact.
    const anchor = "frequently used token-count cache entry"
    countTokens(anchor)
    const cold = "old token-count cache entry"
    countTokens(cold)
    for (let i = 0; i < 2200; i++) {
        countTokens(`unique cache entry ${i}`)
        countTokens(anchor)
    }
    let before = encodes
    countTokens(anchor)
    assert.equal(encodes, before)
    assert.equal(countTokens(cold), upstream.countTokens(cold))
    assert.equal(encodes, before + 1, "entry-count limit evicts cold entries")

    // Fewer than the entry cap, but more than the total character budget.
    countTokens(anchor)
    for (let i = 0; i < 20; i++) countTokens(`${i}:` + "record 中文 ".repeat(25000))
    before = encodes
    countTokens(anchor)
    assert.equal(encodes, before + 1, "character budget also bounds retained text")

    const oversized = "large record ".repeat(25000)
    before = encodes
    const first = countTokens(oversized)
    assert.equal(first, upstream.countTokens(oversized))
    assert.equal(countTokens(oversized), first)
    assert.equal(encodes, before + 2, "oversized strings are counted but not retained")
    assert.equal(constructions, 3, "successful counts share one WASM instance")
})
