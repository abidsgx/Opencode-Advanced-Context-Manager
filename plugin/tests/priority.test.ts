import { describe, it, expect, beforeEach } from "vitest"
import {
    createSessionState,
    getCurrentState,
    setCurrentState,
    buildPriorityMap,
    scoreMessage,
} from "../lib/context/priority"

describe("createSessionState", () => {
    it("creates state with correct sessionId", () => {
        const state = createSessionState("test-session-123")
        expect(state.sessionId).toBe("test-session-123")
        expect(state.currentTurn).toBe(0)
    })

    it("initializes empty maps", () => {
        const state = createSessionState("test")
        expect(state.priority.size).toBe(0)
        expect(state.compressionBlocks.size).toBe(0)
        expect(state.prunedMessages.size).toBe(0)
    })

    it("initializes stats to zero", () => {
        const state = createSessionState("test")
        expect(state.stats.pruneTokenCounter).toBe(0)
        expect(state.stats.totalPruneTokens).toBe(0)
        expect(state.stats.filesPruned).toBe(0)
        expect(state.stats.filesPreserved).toBe(0)
    })

    it("initializes messageIds", () => {
        const state = createSessionState("test")
        expect(state.messageIds.nextRef).toBe(1)
        expect(state.messageIds.byRawId.size).toBe(0)
        expect(state.messageIds.byRef.size).toBe(0)
    })
})

describe("getCurrentState / setCurrentState", () => {
    beforeEach(() => {
        setCurrentState(null)
    })

    it("returns null when no state set", () => {
        expect(getCurrentState()).toBeNull()
    })

    it("returns set state", () => {
        const state = createSessionState("test")
        setCurrentState(state)
        expect(getCurrentState()).toBe(state)
    })

    it("can reset to null", () => {
        setCurrentState(createSessionState("test"))
        setCurrentState(null)
        expect(getCurrentState()).toBeNull()
    })
})

describe("buildPriorityMap", () => {
    it("creates entries from importance scores", () => {
        const map = buildPriorityMap(
            { "src/a.py": 8.0, "src/b.py": 3.0 },
            {},
            {}
        )
        expect(map.size).toBe(2)
        expect(map.get("src/a.py")?.score).toBe(8.0)
        expect(map.get("src/b.py")?.score).toBe(3.0)
    })

    it("adds function scores to file entries", () => {
        const map = buildPriorityMap(
            { "src/a.py": 5.0 },
            { "src/a.py::funcA": 9.0 },
            {}
        )
        // File entry should be updated since function score > file score
        expect(map.get("src/a.py")?.score).toBe(9.0)
    })

    it("blast radius does not decrease score (boost factor <= 1.0)", () => {
        const map = buildPriorityMap(
            { "src/a.py": 8.0, "src/b.py": 5.0 },
            {},
            { "src/a.py": [{ target: "src/b.py", weight: 0.9 }] }
        )
        const bScore = map.get("src/b.py")?.score
        // Boost formula: score * (0.5 + 0.5 * weight) = 5.0 * 0.95 = 4.75
        // Since 4.75 < 5.0, the condition (boosted > original) is false
        // So score stays at 5.0
        expect(bScore).toBe(5.0)
    })

    it("does not boost beyond original if boost is lower", () => {
        const map = buildPriorityMap(
            { "src/a.py": 8.0, "src/b.py": 9.0 },
            {},
            { "src/a.py": [{ target: "src/b.py", weight: 0.1 }] }
        )
        // Low weight blast radius should not increase 9.0
        expect(map.get("src/b.py")?.score).toBe(9.0)
    })
})

describe("scoreMessage", () => {
    it("returns 0 for message with no file references", () => {
        const map = buildPriorityMap({ "src/a.py": 8.0 }, {}, {})
        const msg = { role: "user", content: "Hello world" }
        const result = scoreMessage(msg, map, 7.0)
        expect(result.score).toBe(0)
        expect(result.preserved).toBe(false)
    })

    it("scores message referencing important file", () => {
        const map = buildPriorityMap({ "src/a.py": 8.0 }, {}, {})
        const msg = { role: "assistant", content: "Edited src/a.py to fix the bug" }
        const result = scoreMessage(msg, map, 7.0)
        expect(result.score).toBe(8.0)
    })

    it("preserves message with score above protected threshold", () => {
        const map = buildPriorityMap({ "src/a.py": 8.0 }, {}, {})
        const msg = { role: "assistant", content: "Modified src/a.py" }
        const result = scoreMessage(msg, map, 7.0)
        expect(result.preserved).toBe(true)
    })

    it("does not preserve message below protected threshold", () => {
        const map = buildPriorityMap({ "src/a.py": 5.0 }, {}, {})
        const msg = { role: "assistant", content: "Modified src/a.py" }
        const result = scoreMessage(msg, map, 7.0)
        expect(result.preserved).toBe(false)
    })
})
