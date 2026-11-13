import type { Context } from "@opencode/plugin/promise/plugin"
import type { createEventHandler } from "./hooks"

type V2Event =
    ReturnType<Context["event"]["subscribe"]> extends AsyncIterable<infer Event> ? Event : never

const MAX_PENDING_TOOL_NAMES = 4096

export function createV2EventHandler(
    ctx: Pick<Context, "session">,
    handleEvent: ReturnType<typeof createEventHandler>,
) {
    const toolNames = new Map<string, { sessionID: string; name: string }>()
    const remember = (key: string, sessionID: string, name: string) => {
        toolNames.delete(key)
        if (toolNames.size >= MAX_PENDING_TOOL_NAMES) {
            toolNames.delete(toolNames.keys().next().value!)
        }
        toolNames.set(key, { sessionID, name })
    }

    return async (event: V2Event): Promise<void> => {
        if (event.type === "session.idle" || event.type === "session.deleted") {
            const { sessionID } = event.data
            for (const [key, entry] of toolNames) {
                if (entry.sessionID === sessionID) toolNames.delete(key)
            }
            await handleEvent({ event: { type: event.type, properties: { sessionID } } })
            return
        }
        if (
            event.type !== "session.tool.input.started" &&
            event.type !== "session.tool.called" &&
            event.type !== "session.tool.success" &&
            event.type !== "session.tool.failed"
        ) {
            return
        }
        const { sessionID, assistantMessageID, id } = event.data
        if (typeof assistantMessageID !== "string" || typeof id !== "string") return
        const key = JSON.stringify([sessionID, assistantMessageID, id])
        if (event.type === "session.tool.input.started") {
            if (typeof event.data.name === "string") remember(key, sessionID, event.data.name)
            return
        }

        let name = toolNames.get(key)?.name
        if (name === undefined) {
            // Startup mid-call or eviction: recover once, rather than fetching
            // full history for every called/success/failed event of every tool.
            const messages = await ctx.session.context({ sessionID })
            for (const message of messages as Array<{
                id?: string
                type?: string
                content?: Array<{ type?: string; id?: string; name?: string }>
            }>) {
                if (message.type !== "assistant" || message.id !== assistantMessageID) continue
                name = message.content?.find(
                    (part) =>
                        part &&
                        (part.type === "tool" || part.type === "tool-call") &&
                        part.id === id,
                )?.name
                break
            }
            if (name !== undefined) remember(key, sessionID, name)
        }
        if (event.type !== "session.tool.called") toolNames.delete(key)
        if (name !== "acp_compress") return

        await handleEvent({
            event: {
                type: "message.part.updated",
                time: event.created,
                properties: {
                    part: {
                        type: "tool",
                        tool: name,
                        callID: id,
                        messageID: assistantMessageID,
                        state: {
                            status:
                                event.type === "session.tool.called"
                                    ? "pending"
                                    : event.type === "session.tool.success"
                                      ? "completed"
                                      : "error",
                            time: { completed: event.created },
                        },
                    },
                },
            },
        })
    }
}
