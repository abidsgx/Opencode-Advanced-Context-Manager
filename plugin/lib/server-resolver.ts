import * as fs from "node:fs"
import * as path from "node:path"
import { fileURLToPath } from "node:url"

export interface ResolvedMcpConfig {
    command: string
    args: string[]
    cwd: string
    preferLocal?: boolean
}

function getPluginRoot(): string {
    const __filename = fileURLToPath(import.meta.url)
    const __dirname = path.dirname(__filename)
    return path.resolve(__dirname, "..")
}

function getGlobalServerPath(): string {
    const pluginRoot = getPluginRoot()
    return path.join(pluginRoot, "server")
}

function getLocalServerPath(projectDir: string): string {
    return path.join(projectDir, "server")
}

function findPythonExecutable(): string {
    const candidates = [
        process.env.PYTHON_PATH,
        "python3",
        "python",
        "python.exe",
    ].filter(Boolean) as string[]

    for (const cmd of candidates) {
        try {
            const { spawnSync } = require("node:child_process")
            const result = spawnSync(cmd, ["--version"], { encoding: "utf-8" })
            if (result.status === 0) {
                return cmd
            }
        } catch {
            // Continue to next candidate
        }
    }
    return "python"
}

export function resolveMcpConfig(projectDir: string, userConfig?: Partial<ResolvedMcpConfig>): ResolvedMcpConfig {
    const defaultConfig: ResolvedMcpConfig = {
        command: findPythonExecutable(),
        args: ["-m", "mcp_server"],
        cwd: getGlobalServerPath(),
        preferLocal: true,
    }

    const preferLocal = userConfig?.preferLocal ?? defaultConfig.preferLocal
    const localServerPath = getLocalServerPath(projectDir)
    const globalServerPath = getGlobalServerPath()

    let resolvedCwd = defaultConfig.cwd
    if (preferLocal) {
        if (fs.existsSync(localServerPath)) {
            resolvedCwd = localServerPath
        } else if (fs.existsSync(globalServerPath)) {
            resolvedCwd = globalServerPath
        } else {
            resolvedCwd = "./server"
        }
    } else {
        if (fs.existsSync(globalServerPath)) {
            resolvedCwd = globalServerPath
        } else if (fs.existsSync(localServerPath)) {
            resolvedCwd = localServerPath
        } else {
            resolvedCwd = "./server"
        }
    }

    return {
        command: userConfig?.command ?? defaultConfig.command,
        args: userConfig?.args ?? defaultConfig.args,
        cwd: userConfig?.cwd ?? resolvedCwd,
        preferLocal,
    }
}

export function getServerPaths(projectDir: string): { local: string; global: string } {
    return {
        local: getLocalServerPath(projectDir),
        global: getGlobalServerPath(),
    }
}