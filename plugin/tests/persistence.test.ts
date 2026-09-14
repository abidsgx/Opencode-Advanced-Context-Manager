import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import * as fs from "node:fs"
import * as path from "node:path"
import * as os from "node:os"

vi.mock("../lib/logger", () => ({
    appendDiagnostic: vi.fn(),
}))

describe("persistence", () => {
    let tmpDir: string

    beforeEach(() => {
        tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ctx-test-"))
        // Mock process.cwd to return our temp dir
        vi.stubGlobal("process", { ...process, cwd: () => tmpDir })
    })

    afterEach(() => {
        fs.rmSync(tmpDir, { recursive: true, force: true })
        vi.restoreAllMocks()
    })

    it("saveState creates file and loadState reads it back", async () => {
        const { saveState } = await import("../lib/state/persistence")
        const { loadState } = await import("../lib/state/persistence")

        const state = {
            sessionId: "test-session",
            currentTurn: 3,
            priority: new Map([["src/a.py", { messageId: "src/a.py", score: 8.0, reason: "test" }]]),
            compressionBlocks: new Map(),
            prunedMessages: new Map(),
            messageIds: {
                byRawId: new Map(),
                byRef: new Map(),
                nextRef: 1,
            },
            stats: { pruneTokenCounter: 0, totalPruneTokens: 0, filesPruned: 0, filesPreserved: 0 },
            modelContextLimit: undefined,
            systemPromptTokens: undefined,
        }

        saveState(state as any)

        const loaded = loadState("test-session")
        expect(loaded).not.toBeNull()
        expect(loaded?.sessionId).toBe("test-session")
        expect(loaded?.currentTurn).toBe(3)
        expect(loaded?.priority.get("src/a.py")?.score).toBe(8.0)
    })

    it("loadState returns null for nonexistent session", async () => {
        const { loadState } = await import("../lib/state/persistence")
        const result = loadState("nonexistent-session")
        expect(result).toBeNull()
    })

    it("saveState creates directory if needed", async () => {
        const { saveState } = await import("../lib/state/persistence")

        const state = {
            sessionId: "new-session",
            currentTurn: 0,
            priority: new Map(),
            compressionBlocks: new Map(),
            prunedMessages: new Map(),
            messageIds: { byRawId: new Map(), byRef: new Map(), nextRef: 1 },
            stats: { pruneTokenCounter: 0, totalPruneTokens: 0, filesPruned: 0, filesPreserved: 0 },
            modelContextLimit: undefined,
            systemPromptTokens: undefined,
        }

        saveState(state as any)

        const stateDir = path.join(tmpDir, ".opencode", "ctx-manager", "state")
        expect(fs.existsSync(stateDir)).toBe(true)
    })
})
