import * as fs from "node:fs"
import * as path from "node:path"
import { parse as parseJsonc } from "jsonc-parser"

export interface McpConfig {
    command: string
    args: string[]
    cwd: string
}

export interface ImportanceConfig {
    autoScore: boolean
    threshold: number
    protectedThreshold: number
}

export interface SessionConfig {
    extendedRecording: boolean
    captureEditMetadata: boolean
}

export interface PruningConfig {
    enabled: boolean
    blastRadiusExpansion: boolean
    maxCompressionRatio: number
}

export interface CommandsConfig {
    enabled: boolean
}

export interface PluginConfig {
    enabled: boolean
    debug: boolean
    pruneNotification: "off" | "minimal" | "detailed"
    pruneNotificationType: "chat" | "toast"
    mcp: McpConfig
    importance: ImportanceConfig
    session: SessionConfig
    pruning: PruningConfig
    commands: CommandsConfig
}

const DEFAULT_CONFIG: PluginConfig = {
    enabled: true,
    debug: false,
    pruneNotification: "minimal",
    pruneNotificationType: "toast",
    mcp: {
        command: "python",
        args: ["-m", "mcp_server"],
        cwd: "./server",
    },
    importance: {
        autoScore: true,
        threshold: 3.0,
        protectedThreshold: 7.0,
    },
    session: {
        extendedRecording: true,
        captureEditMetadata: true,
    },
    pruning: {
        enabled: true,
        blastRadiusExpansion: true,
        maxCompressionRatio: 0.3,
    },
    commands: {
        enabled: true,
    },
}

function deepMerge(target: Record<string, unknown>, source: Record<string, unknown>): Record<string, unknown> {
    const result = { ...target }
    for (const key of Object.keys(source)) {
        if (
            source[key] &&
            typeof source[key] === "object" &&
            !Array.isArray(source[key]) &&
            target[key] &&
            typeof target[key] === "object" &&
            !Array.isArray(target[key])
        ) {
            result[key] = deepMerge(target[key] as Record<string, unknown>, source[key] as Record<string, unknown>)
        } else {
            result[key] = source[key]
        }
    }
    return result
}

function findConfigFile(directory: string): string | null {
    const candidates = [
        path.join(directory, "ctx-manager.jsonc"),
        path.join(directory, "ctx-manager.json"),
    ]
    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) return candidate
    }
    return null
}

function findGlobalConfigFile(): string | null {
    const homeDir = process.env.HOME ?? process.env.USERPROFILE
    if (!homeDir) return null

    const candidates = [
        path.join(homeDir, ".config", "opencode", "ctx-manager", "config.jsonc"),
        path.join(homeDir, ".config", "opencode", "ctx-manager", "config.json"),
    ]
    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) return candidate
    }
    return null
}

export async function loadConfig(directory: string): Promise<PluginConfig> {
    let config = { ...DEFAULT_CONFIG }

    const globalConfigPath = findGlobalConfigFile()
    if (globalConfigPath) {
        try {
            const content = fs.readFileSync(globalConfigPath, "utf-8")
            const parsed = parseJsonc(content) as Partial<PluginConfig>
            config = deepMerge(config as Record<string, unknown>, parsed as Record<string, unknown>) as unknown as PluginConfig
        } catch {
            // Ignore invalid global config
        }
    }

    const projectConfigPath = findConfigFile(directory)
    if (projectConfigPath) {
        try {
            const content = fs.readFileSync(projectConfigPath, "utf-8")
            const parsed = parseJsonc(content) as Partial<PluginConfig>
            config = deepMerge(config as Record<string, unknown>, parsed as Record<string, unknown>) as unknown as PluginConfig
        } catch {
            // Ignore invalid project config
        }
    }

    return config
}
