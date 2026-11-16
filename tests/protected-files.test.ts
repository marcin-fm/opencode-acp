import assert from "node:assert/strict"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { test } from "node:test"
import { getConfig } from "../src/config"
import { createCompressRangeTool } from "../src/compress/range"
import { prepareExecutableRangePlans } from "../src/compress/range-utils"
import { buildSearchContext } from "../src/compress/search"
import { createChatMessageTransformHandler } from "../src/v2/hooks"
import { Logger } from "../src/logger"
import { assignMessageRefs } from "../src/message-ids"
import { enforceContextBudget } from "../src/messages/enforce-budget"
import { buildCompressibleRanges } from "../src/messages/inject/utils"
import { prune } from "../src/messages/prune"
import { applyStickyTruncations } from "../src/messages/sticky-truncation"
import { truncateLargeToolOutputs } from "../src/messages/truncate-tools"
import { getFilePathsFromParameters, isFilePathProtected } from "../src/protected-patterns"
import { PromptStore } from "../src/prompts/store"
import { createSessionState, saveSessionState, type SessionStateRegistry } from "../src/state"
import { fromV2SessionMessages, toV2RequestMessages } from "../src/v2/messages"

const policyPattern = "/srv/wikis/*/policies.md"
const policyPath = "/srv/wikis/GLOBAL/policies.md"

function durableTool(id: string, name: string, input: unknown, output: string) {
    return {
        type: "tool",
        id,
        name,
        state: { status: "completed", input, content: [{ type: "text", text: output }] },
    }
}

test("protected path extraction handles V1 and observed V2 tool inputs", () => {
    const patch = [
        "*** Begin Patch",
        "*** Update File: /srv/wikis/GLOBAL/policies.md",
        "*** Move to: /srv/wikis/opencode-acp/policies.md",
        "*** End Patch",
    ].join("\n")
    for (const [tool, input] of [
        ["read", { filePath: policyPath }],
        ["read", { path: policyPath }],
        ["apply_patch", { patchText: patch }],
        ["patch", { patchText: patch }],
    ] as const) {
        assert.equal(
            isFilePathProtected(getFilePathsFromParameters(tool, input), [policyPattern]),
            true,
        )
    }
    assert.ok(
        getFilePathsFromParameters("patch", { patchText: patch }).includes(
            "/srv/wikis/opencode-acp/policies.md",
        ),
    )
    assert.equal(
        isFilePathProtected(
            getFilePathsFromParameters("read", { path: "/srv/wikis/GLOBAL/README.md" }),
            [policyPattern],
        ),
        false,
    )
    assert.deepEqual(getFilePathsFromParameters("read", { path: 123 }), [])
})

test("V1 multiedit paths and existing glob semantics remain supported", () => {
    const paths = getFilePathsFromParameters("multiedit", {
        filePath: "/fixture/ordinary.md",
        edits: [
            { filePath: policyPath },
            { filePath: policyPath },
            { filePath: "/fixture/another.md" },
        ],
    })
    assert.deepEqual(paths, ["/fixture/ordinary.md", policyPath, "/fixture/another.md"])
    assert.equal(isFilePathProtected(paths, [policyPattern]), true)
    assert.equal(isFilePathProtected([policyPath], ["/srv/wikis/*/policies.md"]), true)
    assert.equal(isFilePathProtected([policyPath], ["/srv/*/policies.md"]), false)
    assert.equal(isFilePathProtected([policyPath], ["/srv/**/policies.md"]), true)
    assert.equal(isFilePathProtected(["C:\\work\\policies.md"], ["C:/work/*.md"]), true)
})

test("JSONC file protection configuration is loaded and applied to V2 reads", async () => {
    const sandbox = await mkdtemp(join(tmpdir(), "acp-protected-config-"))
    const previous = process.env.OPENCODE_CONFIG_DIR
    try {
        await writeFile(
            join(sandbox, "acp.jsonc"),
            `{
                // Preserve the configured file patterns.
                "protectedFilePatterns": ["/fixture/*/policies.md"],
            }`,
        )
        process.env.OPENCODE_CONFIG_DIR = sandbox
        const config = getConfig({ directory: sandbox })
        assert.ok(config.protectedFilePatterns.includes("/fixture/*/policies.md"))
        assert.equal(
            isFilePathProtected(
                getFilePathsFromParameters("read", { path: "/fixture/project/policies.md" }),
                config.protectedFilePatterns,
            ),
            true,
        )
    } finally {
        if (previous === undefined) delete process.env.OPENCODE_CONFIG_DIR
        else process.env.OPENCODE_CONFIG_DIR = previous
        await rm(sandbox, { recursive: true, force: true })
    }
})

test("V1 and V2 policy reads and patches stay visible after range compression", async () => {
    const sandbox = await mkdtemp(join(tmpdir(), "acp-protected-files-"))
    const keys = ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "OPENCODE_CONFIG_DIR"] as const
    const previous = keys.map((key) => process.env[key])
    for (const key of keys) process.env[key] = sandbox
    try {
        const logger = new Logger(false, "silent")
        const config = getConfig({ directory: sandbox })
        config.protectedFilePatterns = [policyPattern]
        config.compress.protectedTools = ["acp_compress"]
        config.compress.minCompressRange = 0
        config.qualityGate.enabled = false
        config.pruneNotification = "off"
        const policyOutput = "Global policy content to retain verbatim."
        const patchOutput = "Updated policy file."
        const legacyReadOutput = "V1 policy read to retain verbatim."
        const legacyPatchOutput = "V1 policy patch to retain verbatim."
        const durable = [
            { id: "user", type: "user", text: "Read files." },
            { id: "before", type: "assistant", content: [{ type: "text", text: "Before read." }] },
            {
                id: "policy",
                type: "assistant",
                content: [durableTool("policy-call", "read", { path: policyPath }, policyOutput)],
            },
            {
                id: "ordinary",
                type: "assistant",
                content: [
                    durableTool(
                        "ordinary-call",
                        "read",
                        { path: "/srv/wikis/GLOBAL/README.md" },
                        "Ordinary output.",
                    ),
                ],
            },
            {
                id: "patch",
                type: "assistant",
                content: [
                    durableTool(
                        "patch-call",
                        "patch",
                        {
                            patchText:
                                "*** Begin Patch\n*** Update File: /srv/wikis/opencode-acp/policies.md\n*** End Patch",
                        },
                        patchOutput,
                    ),
                ],
            },
            {
                id: "legacy-read",
                type: "assistant",
                content: [
                    durableTool(
                        "legacy-read-call",
                        "read",
                        { filePath: policyPath },
                        legacyReadOutput,
                    ),
                ],
            },
            {
                id: "legacy-patch",
                type: "assistant",
                content: [
                    durableTool(
                        "legacy-patch-call",
                        "apply_patch",
                        {
                            patchText:
                                "*** Begin Patch\n*** Update File: /srv/wikis/opencode-acp/policies.md\n*** End Patch",
                        },
                        legacyPatchOutput,
                    ),
                ],
            },
            { id: "after", type: "assistant", content: [{ type: "text", text: "After read." }] },
        ]
        const messages = fromV2SessionMessages(durable, "session")
        const state = createSessionState()
        state.sessionId = "session"
        state.storageDir = sandbox
        assignMessageRefs(state, messages)
        const ranges = buildCompressibleRanges(
            messages,
            state,
            config.compress.protectedTools,
            config.protectedFilePatterns,
        )
        assert.deepEqual(
            ranges.protected.map((range) => [range.startRef, range.endRef, range.tools]),
            [
                ["m00003", "m00003", ["read"]],
                ["m00005", "m00007", ["patch", "read", "apply_patch"]],
            ],
        )

        const input = {
            topic: "File inspection",
            dangerous: true,
            content: [{ startId: "m00002", endId: "m00008", summary: "Inspected ordinary files." }],
        }
        const planned = prepareExecutableRangePlans(
            input,
            buildSearchContext(state, messages),
            state,
            config,
            logger,
        )
        assert.deepEqual(planned.protectedMessageIds, [
            "policy",
            "patch",
            "legacy-read",
            "legacy-patch",
        ])
        assert.deepEqual(planned.plans[0].selection.messageIds, ["before", "ordinary", "after"])

        const tool = createCompressRangeTool({
            client: { session: { context: async () => durable } },
            registry: { get: () => state } as unknown as SessionStateRegistry,
            logger,
            config,
            prompts: new PromptStore(logger, sandbox),
            assertPermission: async () => {},
        })
        const result = await tool.execute(input, {
            sessionID: "session",
            messageID: "compress",
            id: "compress-call",
        })
        assert.match(result.content, /Compressed 3 messages/)
        assert.match(result.content, /4 requested message\(s\) were NOT compressed/)
        const visible = fromV2SessionMessages(durable, "session")
        prune(state, logger, config, visible)
        assert.deepEqual(
            visible.map((message) => message.info.id),
            ["user", "policy", "patch", "legacy-read", "legacy-patch"],
        )
        const wire = JSON.stringify(toV2RequestMessages(visible))
        assert.ok(wire.includes(policyOutput))
        assert.ok(wire.includes(patchOutput))
        assert.ok(wire.includes(legacyReadOutput))
        assert.ok(wire.includes(legacyPatchOutput))
        assert.ok(!wire.includes("Ordinary output."))
    } finally {
        keys.forEach((key, i) => {
            if (previous[i] === undefined) delete process.env[key]
            else process.env[key] = previous[i]
        })
        await rm(sandbox, { recursive: true, force: true })
    }
})

test("emergency truncation and sticky replay respect protected messages", async () => {
    const sandbox = await mkdtemp(join(tmpdir(), "acp-protected-emergency-"))
    const keys = ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "OPENCODE_CONFIG_DIR"] as const
    const previous = keys.map((key) => process.env[key])
    for (const key of keys) process.env[key] = sandbox
    try {
        const logger = new Logger(false, "silent")
        const warnings: string[] = []
        logger.warn = (message) => warnings.push(message)
        const config = getConfig({ directory: sandbox })
        config.protectedFilePatterns = [policyPattern]
        config.compress.protectedTools = ["special_*", "acp_compress"]
        config.compress.completionReserveTokens = 19000
        config.gc.majorGcThresholdPercent = 500
        const state = createSessionState()
        state.modelContextLimit = 20000
        const longOutput = "alpha beta gamma delta 1234 example content\n".repeat(1000)
        const durable = [
            { id: "user", type: "user", text: "Start." },
            {
                id: "policy",
                type: "assistant",
                content: [
                    durableTool("policy-call", "read", { path: policyPath }, longOutput),
                    durableTool("sibling-call", "read", { path: "/ordinary" }, longOutput),
                ],
            },
            {
                id: "wildcard",
                type: "assistant",
                content: [durableTool("wildcard-call", "special_tool", {}, longOutput)],
            },
            {
                id: "ordinary",
                type: "assistant",
                content: [durableTool("ordinary-call", "read", { path: "/ordinary" }, longOutput)],
            },
            ...[1, 2, 3].map((n) => ({
                id: `recent-${n}`,
                type: "assistant",
                content: [{ type: "text", text: `Recent ${n}` }],
            })),
        ]
        const messages = fromV2SessionMessages(durable, "session")
        state.stickyTruncations = {
            "policy-call": "old policy truncation",
            "sibling-call": "old sibling truncation",
            "wildcard-call": "old protected-tool truncation",
            "ordinary-call": "old ordinary truncation",
        }
        applyStickyTruncations(
            state,
            messages,
            logger,
            config.compress.protectedTools,
            config.protectedFilePatterns,
        )
        assert.equal(messages[1].parts[0].state.output, longOutput)
        assert.equal(messages[1].parts[1].state.output, longOutput)
        assert.equal(messages[2].parts[0].state.output, longOutput)
        assert.equal(messages[3].parts[0].state.output, "old ordinary truncation")

        const fresh = fromV2SessionMessages(durable, "session")
        truncateLargeToolOutputs(state, config, logger, fresh)
        assert.equal(fresh[1].parts[0].state.output, longOutput)
        assert.equal(fresh[1].parts[1].state.output, longOutput)
        assert.equal(fresh[2].parts[0].state.output, longOutput)
        assert.notEqual(fresh[3].parts[0].state.output, longOutput)

        const budgetMessages = fromV2SessionMessages(durable, "session")
        const result = enforceContextBudget(state, config, logger, budgetMessages)
        assert.equal(budgetMessages[1].parts[0].state.output, longOutput)
        assert.equal(budgetMessages[1].parts[1].state.output, longOutput)
        assert.equal(budgetMessages[2].parts[0].state.output, longOutput)
        assert.notEqual(budgetMessages[3].parts[0].state.output, longOutput)
        assert.ok(result && result.finalEstimate > result.budget)
        assert.ok(warnings.some((message) => message.includes("still over budget")))
    } finally {
        keys.forEach((key, i) => {
            if (previous[i] === undefined) delete process.env[key]
            else process.env[key] = previous[i]
        })
        await rm(sandbox, { recursive: true, force: true })
    }
})

test("real message hook applies model-specific protection to emergency truncation", async () => {
    const sandbox = await mkdtemp(join(tmpdir(), "acp-protected-hook-"))
    const keys = ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "OPENCODE_CONFIG_DIR"] as const
    const previous = keys.map((key) => process.env[key])
    for (const key of keys) process.env[key] = sandbox
    const logger = new Logger(false, "silent")
    const state = createSessionState()
    state.sessionId = "protected-hook"
    state.storageDir = sandbox
    try {
        const config = getConfig({ directory: sandbox })
        config.protectedFilePatterns = [policyPattern]
        config.compress.protectedTools = ["acp_compress"]
        config.compress.emergencyTruncation = false
        config.compress.completionReserveTokens = 19000
        config.compress.maxContextLimit = Number.MAX_SAFE_INTEGER
        config.compress.minContextLimit = Number.MAX_SAFE_INTEGER
        config.compress.iterationNudgeThreshold = Number.MAX_SAFE_INTEGER
        config.compress.images.enabled = false
        config.compress.providers = {
            provider: {
                models: {
                    model: {
                        emergencyTruncation: true,
                        protectedTools: ["special_*", "acp_compress"],
                    },
                },
            },
        }
        config.gc.majorGcThresholdPercent = 500
        const registry = {
            async getOrCreate() {
                return state
            },
            resolveModelLimit() {
                return 20000
            },
        } as unknown as SessionStateRegistry
        const handler = createChatMessageTransformHandler(
            {},
            registry,
            logger,
            config,
            new PromptStore(logger, sandbox),
            { global: undefined, agents: {} },
        )
        const longOutput = "alpha beta gamma delta 1234 example content\n".repeat(1000)
        const durable = [
            {
                id: "user",
                type: "user",
                text: "Start.",
                model: { providerID: "provider", id: "model" },
            },
            {
                id: "policy",
                type: "assistant",
                content: [durableTool("policy-call", "read", { path: policyPath }, longOutput)],
            },
            {
                id: "wildcard",
                type: "assistant",
                content: [durableTool("wildcard-call", "special_tool", {}, longOutput)],
            },
            {
                id: "ordinary",
                type: "assistant",
                content: [durableTool("ordinary-call", "read", { path: "/ordinary" }, longOutput)],
            },
            ...[1, 2, 3].map((n) => ({
                id: `recent-${n}`,
                type: "assistant",
                content: [{ type: "text", text: `Recent ${n}` }],
            })),
        ]
        state.stickyTruncations = {
            "policy-call": "old policy truncation",
            "wildcard-call": "old protected-tool truncation",
        }
        const outputFor = (messages: ReturnType<typeof fromV2SessionMessages>, callID: string) => {
            const part = messages
                .flatMap((message) => message.parts)
                .find((candidate) => candidate.type === "tool" && candidate.callID === callID)
            assert.ok(part && part.type === "tool")
            return part.state.output
        }
        const first = { messages: fromV2SessionMessages(durable, "protected-hook") }
        await handler({}, first)
        assert.ok(outputFor(first.messages, "policy-call").startsWith(longOutput))
        assert.ok(outputFor(first.messages, "wildcard-call").startsWith(longOutput))
        assert.ok(!outputFor(first.messages, "ordinary-call").startsWith(longOutput))
        assert.equal(state.stickyTruncations["policy-call"], "old policy truncation")
        assert.equal(state.stickyTruncations["wildcard-call"], "old protected-tool truncation")
        assert.ok(state.stickyTruncations["ordinary-call"])

        // Protection is evaluated on each request, without erasing old frozen state.
        config.protectedFilePatterns = []
        config.compress.providers.provider.models!.model.protectedTools = ["acp_compress"]
        const second = { messages: fromV2SessionMessages(durable, "protected-hook") }
        await handler({}, second)
        assert.ok(outputFor(second.messages, "policy-call").startsWith("old policy truncation"))
        assert.ok(
            outputFor(second.messages, "wildcard-call").startsWith("old protected-tool truncation"),
        )
        assert.ok(!outputFor(second.messages, "ordinary-call").startsWith(longOutput))
    } finally {
        await saveSessionState(state, logger)
        keys.forEach((key, i) => {
            if (previous[i] === undefined) delete process.env[key]
            else process.env[key] = previous[i]
        })
        await rm(sandbox, { recursive: true, force: true })
    }
})
