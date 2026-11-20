import type { PluginConfig } from "./config"
import { type HostPermissionSnapshot, resolveEffectiveCompressPermission } from "./host-permissions"
import type { SessionState, WithParts } from "./state"
import { getRequestInput } from "./messages/query"

export const compressPermission = (
    state: SessionState,
    config: PluginConfig,
): "ask" | "allow" | "deny" => {
    return state.compressPermission ?? config.compress.permission
}

export const syncCompressPermissionState = (
    state: SessionState,
    config: PluginConfig,
    hostPermissions: HostPermissionSnapshot,
    messages: WithParts[],
): void => {
    const activeAgent = getRequestInput(messages)?.info.agent
    state.compressPermission = resolveEffectiveCompressPermission(
        config.compress.permission,
        hostPermissions,
        activeAgent,
    )
}
