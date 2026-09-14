import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import * as fs from "node:fs"
import * as path from "node:path"

vi.mock("../lib/logger", () => ({
    appendDiagnostic: vi.fn(),
}))

describe("config", () => {
    let loadConfig: typeof import("../lib/config").loadConfig

    beforeEach(async () => {
        vi.resetModules()
        const mod = await import("../lib/config")
        loadConfig = mod.loadConfig
    })

    it("returns default config when no files exist", async () => {
        const config = await loadConfig("/nonexistent/path")
        expect(config.enabled).toBe(true)
        expect(config.debug).toBe(false)
        expect(config.importance.autoScore).toBe(true)
        expect(config.importance.threshold).toBe(3.0)
        expect(config.importance.protectedThreshold).toBe(7.0)
        expect(config.pruning.enabled).toBe(true)
        expect(config.pruning.maxCompressionRatio).toBe(0.3)
        expect(config.commands.enabled).toBe(true)
    })

    it("has correct default MCP config", async () => {
        const config = await loadConfig("/nonexistent/path")
        expect(config.mcp.command).toBe("python")
        expect(config.mcp.args).toEqual(["-m", "mcp_server"])
    })

    it("has correct default session config", async () => {
        const config = await loadConfig("/nonexistent/path")
        expect(config.session.extendedRecording).toBe(true)
        expect(config.session.captureEditMetadata).toBe(true)
    })

    it("has valid pruneNotificationType", async () => {
        const config = await loadConfig("/nonexistent/path")
        expect(["off", "minimal", "detailed"]).toContain(config.pruneNotification)
        expect(["chat", "toast"]).toContain(config.pruneNotificationType)
    })
})
