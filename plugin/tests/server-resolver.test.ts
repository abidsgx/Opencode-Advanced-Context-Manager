import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { resolveMcpConfig, getServerPaths } from "../lib/server-resolver"
import { loadConfig } from "../lib/config"

/**
 * `fs.existsSync` is the resolver's only link to the filesystem, so it is the seam
 * we control here. Paths not listed in `forced` fall through to the real check, which
 * lets the loadConfig cases below use a real temp project directory.
 */
const forced = new Map<string, boolean>()

vi.mock("node:fs", async (importOriginal) => {
    const actual = await importOriginal<typeof import("node:fs")>()
    return {
        ...actual,
        existsSync: (target: fs.PathLike) => {
            const key = String(target)
            if (forced.has(key)) return forced.get(key) as boolean
            return actual.existsSync(target)
        },
    }
})

const projectDir = path.join(os.tmpdir(), "ctx-manager-resolver-fixture")

function serverPaths() {
    return getServerPaths(projectDir)
}

let tempConfigDir: string

beforeEach(() => {
    forced.clear()
    tempConfigDir = fs.mkdtempSync(path.join(os.tmpdir(), "ctx-manager-config-"))
})

afterEach(() => {
    fs.rmSync(tempConfigDir, { recursive: true, force: true })
})

describe("getServerPaths", () => {
    it("returns an absolute local path under the project directory", () => {
        const { local } = serverPaths()
        expect(path.isAbsolute(local)).toBe(true)
        expect(local).toBe(path.join(projectDir, "server"))
    })

    it("returns the packaged server inside the plugin directory as global", () => {
        const { global } = serverPaths()
        expect(path.isAbsolute(global)).toBe(true)
        expect(path.basename(global)).toBe("server")
        // <pluginRoot>/server — pluginRoot is the parent of lib/
        expect(path.basename(path.dirname(global))).toBe("plugin")
    })

    it("never returns the same path for local and global in a foreign project", () => {
        const { local, global } = serverPaths()
        expect(local).not.toBe(global)
    })
})

describe("resolveMcpConfig: server path precedence", () => {
    it("prefers the project-local server when it exists", () => {
        forced.set(serverPaths().local, true)
        forced.set(serverPaths().global, true)
        expect(resolveMcpConfig(projectDir).cwd).toBe(serverPaths().local)
    })

    it("falls back to the packaged global server when the project has none", () => {
        forced.set(serverPaths().local, false)
        forced.set(serverPaths().global, true)
        expect(resolveMcpConfig(projectDir).cwd).toBe(serverPaths().global)
    })

    it("keeps the legacy relative default when neither server exists", () => {
        forced.set(serverPaths().local, false)
        forced.set(serverPaths().global, false)
        // Documented last-resort value; resolving it against the host cwd is what
        // produced `spawn python ENOENT` before this feature existed.
        expect(resolveMcpConfig(projectDir).cwd).toBe("./server")
    })

    it("returns an absolute cwd whenever a server directory is found", () => {
        for (const localExists of [true, false]) {
            forced.set(serverPaths().local, localExists)
            forced.set(serverPaths().global, true)
            expect(path.isAbsolute(resolveMcpConfig(projectDir).cwd)).toBe(true)
        }
    })

    it("defaults args to `-m mcp_server`", () => {
        forced.set(serverPaths().local, true)
        expect(resolveMcpConfig(projectDir).args).toEqual(["-m", "mcp_server"])
    })

    it("resolves paths independently of the process cwd", () => {
        forced.set(serverPaths().local, true)
        const before = process.cwd()
        try {
            process.chdir(os.tmpdir())
            expect(resolveMcpConfig(projectDir).cwd).toBe(serverPaths().local)
        } finally {
            process.chdir(before)
        }
    })
})

describe("resolveMcpConfig: explicit user configuration", () => {
    it("lets user command, args and cwd win over detection", () => {
        forced.set(serverPaths().local, true)
        const resolved = resolveMcpConfig(projectDir, {
            command: "C:/custom/python.exe",
            args: ["-m", "other_server"],
            cwd: "C:/custom/server",
        })
        expect(resolved).toEqual({
            command: "C:/custom/python.exe",
            args: ["-m", "other_server"],
            cwd: "C:/custom/server",
            preferLocal: true,
        })
    })

    it("falls back to detection for fields the user did not set", () => {
        forced.set(serverPaths().local, true)
        const resolved = resolveMcpConfig(projectDir, { command: "C:/custom/python.exe" })
        expect(resolved.command).toBe("C:/custom/python.exe")
        expect(resolved.cwd).toBe(serverPaths().local)
    })

    it("always returns a non-empty command so spawn() gets a candidate", () => {
        forced.set(serverPaths().local, false)
        forced.set(serverPaths().global, false)
        expect(resolveMcpConfig(projectDir).command).toMatch(/\S/)
    })
})

describe("loadConfig: auto-detection wiring", () => {
    it("auto-detects by default and points cwd at the project-local server", () => {
        forced.set(path.join(tempConfigDir, "server"), true)
        const config = loadConfig(tempConfigDir)
        return expect(config).resolves.toMatchObject({
            mcp: { cwd: path.join(tempConfigDir, "server") },
        })
    })

    it("uses the packaged global server for a project without a server directory", () => {
        forced.set(path.join(tempConfigDir, "server"), false)
        const expected = getServerPaths(tempConfigDir).global
        return expect(loadConfig(tempConfigDir)).resolves.toMatchObject({
            mcp: { cwd: expected },
        })
    })

    it("preserves an explicit cwd when autoDetect is disabled", () => {
        fs.writeFileSync(
            path.join(tempConfigDir, "ctx-manager.jsonc"),
            JSON.stringify({ mcp: { cwd: "C:/pinned/server", autoDetect: false } })
        )
        return expect(loadConfig(tempConfigDir)).resolves.toMatchObject({
            mcp: { cwd: "C:/pinned/server" },
        })
    })

    it("documents current behaviour: autoDetect replaces an explicit cwd", () => {
        // loadConfig forwards only command/args into the resolver, so a cwd from the
        // user's config file is overwritten by detection while autoDetect is on.
        // autoDetect:false (previous case) is the supported escape hatch. If this
        // assertion ever fails, the contract changed — update the smoke test.
        fs.writeFileSync(
            path.join(tempConfigDir, "ctx-manager.jsonc"),
            JSON.stringify({ mcp: { cwd: "C:/pinned/server" } })
        )
        forced.set(path.join(tempConfigDir, "server"), true)
        return expect(loadConfig(tempConfigDir)).resolves.toMatchObject({
            mcp: { cwd: path.join(tempConfigDir, "server") },
        })
    })

    it("keeps auto-detected mcp fields complete", async () => {
        forced.set(path.join(tempConfigDir, "server"), true)
        const config = await loadConfig(tempConfigDir)
        expect(config.mcp.command).toMatch(/\S/)
        expect(config.mcp.args).toEqual(["-m", "mcp_server"])
        expect(config.mcp.autoDetect).toBe(true)
        expect(config.mcp.preferLocal).toBe(true)
    })
})
