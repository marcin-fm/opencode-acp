import assert from "node:assert/strict"
import { test } from "node:test"
import { getInvalidConfigKeys, validateConfigTypes } from "../src/config-validation"

test("recognizes the compression configuration section and its model overrides", () => {
    const config = {
        compress: {
            maxContextLimit: "70%",
            emergencyThresholdPercent: "80%",
            providers: {
                openai: {
                    models: {
                        "gpt-6.2-sol": {
                            maxContextLimit: 120000,
                            emergencyThresholdPercent: 300000,
                        },
                    },
                },
            },
        },
    }

    assert.deepEqual(getInvalidConfigKeys(config), [])
    assert.deepEqual(validateConfigTypes(config), [])
})

test("reports unknown compression keys and names the section in type errors", () => {
    assert.deepEqual(
        getInvalidConfigKeys({
            acp_compress: {},
            compress: { maxContextLmit: 120000 },
        }),
        ["acp_compress", "compress.maxContextLmit"],
    )
    assert.deepEqual(validateConfigTypes({ compress: true }), [
        { key: "compress", expected: "object", actual: "boolean" },
    ])
})
