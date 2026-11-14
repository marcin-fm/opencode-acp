import assert from "node:assert/strict"
import { test } from "node:test"
import { setImmediate } from "node:timers/promises"
import { currentDiagnostic, diagnoseContext, type TokenCacheDiagnostic } from "../src/diagnostics"
import { Logger } from "../src/logger"
import { countTokens } from "../src/context-usage"
import { parseDiagnosticLine } from "../scripts/diagnostics-summary"

function capture() {
    const records: Array<{
        sessionId: string
        status: string
        tokenCache?: TokenCacheDiagnostic
    }> = []
    const logger = new Logger(false, "info")
    logger.diagnostic = async (_event, data) => {
        const line = `2026-11-14T00:00:00.000Z INFO diagnostics: context | ${JSON.stringify({ ...data, version: "fixture" })} | v=fixture`
        const parsed = parseDiagnosticLine(line)
        assert.equal(parsed.kind, "diagnostic")
        if (parsed.kind === "diagnostic") records.push(parsed.record)
    }
    return { logger, records }
}

test("interleaved requests attribute cache work to their own scope, not global deltas", async () => {
    const { logger, records } = capture()
    const own = "request A private text 中文"
    const shared = "shared request text 中文"
    await Promise.all([
        diagnoseContext(logger, "a", async () => {
            countTokens(own)
            countTokens(own)
            await setImmediate()
            countTokens(shared)
        }),
        diagnoseContext(logger, "b", async () => {
            countTokens(shared)
            await setImmediate()
            countTokens(shared)
            countTokens(shared)
        }),
        Promise.resolve().then(() => countTokens("unscoped work must not be attributed")),
    ])
    const a = records.find((record) => record.sessionId === "a")!.tokenCache!
    const b = records.find((record) => record.sessionId === "b")!.tokenCache!
    for (const counters of [a, b]) {
        assert.equal(counters.hits, 2)
        assert.equal(counters.misses, 1)
        assert.equal(counters.evictions, 0)
        assert.equal(counters.failures, 0)
        assert.ok(counters.countMs >= 0)
        assert.equal(counters.capacity, 65536)
    }
    assert.equal(a.inputChars, own.length * 2 + shared.length)
    assert.equal(a.missChars, own.length)
    assert.equal(b.inputChars, shared.length * 3)
    assert.equal(b.missChars, shared.length)
    assert.ok(!JSON.stringify(records).includes(own))
    assert.ok(!JSON.stringify(records).includes(shared))
})

test("nested scopes, failed requests and late continuations preserve attribution", async () => {
    const { logger, records } = capture()
    let resume!: () => void
    const barrier = new Promise<void>((resolve) => (resume = resolve))
    let late!: Promise<void>
    await diagnoseContext(logger, "outer", async () => {
        countTokens("outer count")
        await diagnoseContext(logger, "inner", async () => {
            countTokens("inner count")
            countTokens("inner count")
        })
        countTokens("outer count")
        late = barrier.then(() => {
            assert.equal(currentDiagnostic(), undefined)
            countTokens("late count")
        })
    })
    const snapshot = JSON.stringify(records)
    resume()
    await late
    assert.equal(JSON.stringify(records), snapshot, "finished records cannot gain late counts")
    for (const record of records) {
        assert.equal(record.tokenCache!.misses, 1)
        assert.equal(record.tokenCache!.hits, 1)
    }

    const original = new Error("synthetic request failure")
    await assert.rejects(
        diagnoseContext(logger, "failed", async () => {
            countTokens("count before request failure")
            throw original
        }),
        (error) => error === original,
    )
    assert.equal(records.at(-1)!.status, "error")
    assert.equal(records.at(-1)!.tokenCache!.misses, 1)
    assert.equal(records.at(-1)!.tokenCache!.failures, 0, "only counting failures are counted")
})

test("empty counts stay unmeasured and silent diagnostics remain disabled", async () => {
    const { logger, records } = capture()
    await diagnoseContext(logger, "empty", async () => assert.equal(countTokens(""), 0))
    assert.equal(records[0].tokenCache, undefined)
    const silent = new Logger(false, "silent")
    silent.diagnostic = async () => assert.fail("silent scope must not emit a record")
    await diagnoseContext(silent, "silent", async (diagnostic) => {
        assert.equal(diagnostic, undefined)
        assert.equal(currentDiagnostic(), undefined)
        assert.ok(countTokens("silent count") > 0)
    })
})
