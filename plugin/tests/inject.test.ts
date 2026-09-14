import { describe, it, expect } from "vitest"
import {
    buildSystemPromptExtension,
    buildPruningNotification,
    injectMetadata,
} from "../lib/context/inject"
import type { PluginConfig } from "../lib/config"

const defaultConfig: PluginConfig = {
    enabled: true,
    debug: false,
    pruneNotification: "minimal",
    pruneNotificationType: "toast",
    mcp: { command: "python", args: [], cwd: "." },
    importance: { autoScore: true, threshold: 3.0, protectedThreshold: 7.0 },
    session: { extendedRecording: true, captureEditMetadata: true },
    pruning: { enabled: true, blastRadiusExpansion: true, maxCompressionRatio: 0.3 },
    commands: { enabled: true },
}

describe("buildSystemPromptExtension", () => {
    it("includes ctx-manager tags", () => {
        const result = buildSystemPromptExtension({}, {}, [], null)
        expect(result).toContain("<ctx-manager>")
        expect(result).toContain("</ctx-manager>")
    })

    it("includes goal when provided", () => {
        const result = buildSystemPromptExtension({}, {}, [], "Fix auth bug")
        expect(result).toContain("## Current Goal: Fix auth bug")
    })

    it("includes top file scores", () => {
        const scores = { "src/a.py": 9.0, "src/b.py": 5.0 }
        const result = buildSystemPromptExtension(scores, {}, [], null)
        expect(result).toContain("src/a.py: 9.0")
        expect(result).toContain("File Importance Scores")
    })

    it("limits to top 10 files", () => {
        const scores: Record<string, number> = {}
        for (let i = 0; i < 15; i++) {
            scores[`src/file_${i}.py`] = i
        }
        const result = buildSystemPromptExtension(scores, {}, [], null)
        // Should contain only 10 file lines
        const fileLines = result.split("\n").filter((l) => l.startsWith("  - src/file_"))
        expect(fileLines.length).toBe(10)
    })

    it("includes function scores", () => {
        const funcScores = { "src/a.py::authenticate": 8.5 }
        const result = buildSystemPromptExtension({}, funcScores, [], null)
        expect(result).toContain("src/a.py::authenticate: 8.5")
        expect(result).toContain("Function Importance Scores")
    })

    it("includes high-weight connections", () => {
        const connections = [
            { from: "src/a.py", to: "src/b.py", weight: 0.8, relation: "calls" },
            { from: "src/c.py", to: "src/d.py", weight: 0.2, relation: "imports" },
        ]
        const result = buildSystemPromptExtension({}, {}, connections, null)
        expect(result).toContain("src/a.py → src/b.py")
        expect(result).not.toContain("src/c.py → src/d.py") // weight < 0.5
    })

    it("returns empty sections when no data", () => {
        const result = buildSystemPromptExtension({}, {}, [], null)
        expect(result).toContain("<ctx-manager>")
        expect(result).not.toContain("File Importance Scores")
        expect(result).not.toContain("Function Importance Scores")
    })
})

describe("buildPruningNotification", () => {
    it("includes pruned count", () => {
        const result = buildPruningNotification(["src/a.py"], [], 500)
        expect(result).toContain("Pruned 1")
        expect(result).toContain("500 tokens")
    })

    it("includes preserved count", () => {
        const result = buildPruningNotification([], ["src/b.py"], 0)
        expect(result).toContain("Preserved 1")
    })

    it("wraps in notification tags", () => {
        const result = buildPruningNotification([], [], 0)
        expect(result).toContain("<ctx-manager-notification>")
        expect(result).toContain("</ctx-manager-notification>")
    })
})

describe("injectMetadata", () => {
    it("returns original prompt when disabled", () => {
        const config = { ...defaultConfig, enabled: false }
        const result = injectMetadata("System prompt", {}, {}, [], null, config)
        expect(result).toBe("System prompt")
    })

    it("appends extension when enabled", () => {
        const result = injectMetadata("System prompt", { "src/a.py": 8.0 }, {}, [], null, defaultConfig)
        expect(result).toContain("System prompt")
        expect(result).toContain("<ctx-manager>")
        expect(result).toContain("src/a.py: 8.0")
    })
})
