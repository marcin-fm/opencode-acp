import type { WithParts } from "../state"
import { fromV2SessionMessages } from "./messages"

interface ForkSource {
    sessionID: string
    boundary: { type: "before" | "through"; messageID: string }
}

interface SessionLineage {
    parentID?: string
    fork?: ForkSource
}

interface SessionReader {
    session?: {
        get?: (input: { sessionID: string }) => Promise<SessionLineage>
        context?: (input: { sessionID: string }) => Promise<unknown>
    }
}

/** A V2 subagent has parentID; a V2 fork has a separate source and copy boundary. */
export async function getV2SessionLineage(
    client: unknown,
    sessionID: string,
): Promise<SessionLineage> {
    try {
        const info = await (client as SessionReader).session?.get?.({ sessionID })
        const fork = info?.fork
        return {
            parentID: typeof info?.parentID === "string" ? info.parentID : undefined,
            fork:
                typeof fork?.sessionID === "string" &&
                typeof fork.boundary?.messageID === "string" &&
                (fork.boundary.type === "before" || fork.boundary.type === "through")
                    ? fork
                    : undefined,
        }
    } catch {
        return {}
    }
}

/** Project only the durable source prefix OpenCode could have copied. */
export async function fetchV2ForkSourceMessages(
    client: unknown,
    { sessionID, boundary }: ForkSource,
): Promise<WithParts[]> {
    const history = await (client as SessionReader).session?.context?.({ sessionID })
    if (!Array.isArray(history)) return []
    const end = history.findIndex((row) => row?.id === boundary.messageID)
    // A later native compaction can make the original boundary unavailable.
    // Do not infer it from a newer, potentially different transcript window.
    if (end === -1) return []
    const copied = history
        .slice(0, end + (boundary.type === "through" ? 1 : 0))
        .filter((row) => row?.type !== "assistant" || typeof row.time?.completed === "number")
    return fromV2SessionMessages(copied, sessionID)
}
