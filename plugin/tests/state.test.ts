import { describe, it, expect, vi, beforeEach } from "vitest"

vi.mock("../lib/logger", () => ({
    appendDiagnostic: vi.fn(),
}))

vi.mock("../lib/state/persistence", () => ({
    loadState: vi.fn().mockReturnValue(null),
    saveState: vi.fn(),
}))

import {
    initSession,
    getSession,
    getSessionId,
    resetSession,
    updateSession,
    recordPrunedTokens,
    incrementTurn,
} from "../lib/state/state"
import { loadState, saveState } from "../lib/state/persistence"

beforeEach(() => {
    vi.clearAllMocks()
    resetSession()
    vi.mocked(loadState).mockReturnValue(null)
})

describe("initSession", () => {
    it("creates new session state", () => {
        const state = initSession("session-123")
        expect(state.sessionId).toBe("session-123")
        expect(getSessionId()).toBe("session-123")
    })

    it("returns same state for same session", () => {
        const state1 = initSession("session-123")
        const state2 = initSession("session-123")
        expect(state1).toBe(state2)
    })

    it("loads existing state if available", () => {
        const existing = {
            sessionId: "session-123",
            currentTurn: 5,
            priority: new Map(),
            compressionBlocks: new Map(),
            prunedMessages: new Map(),
            messageIds: { byRawId: new Map(), byRef: new Map(), nextRef: 3 },
            stats: { pruneTokenCounter: 0, totalPruneTokens: 0, filesPruned: 0, filesPreserved: 0 },
        }
        vi.mocked(loadState).mockReturnValue(existing as any)
        const state = initSession("session-123")
        expect(state.currentTurn).toBe(5)
    })

    it("saves new state", () => {
        initSession("session-123")
        expect(saveState).toHaveBeenCalled()
    })
})

describe("getSession / getSessionId", () => {
    it("returns null before init", () => {
        expect(getSession()).toBeNull()
        expect(getSessionId()).toBeNull()
    })

    it("returns state after init", () => {
        initSession("test")
        expect(getSession()).not.toBeNull()
        expect(getSessionId()).toBe("test")
    })
})

describe("resetSession", () => {
    it("clears state and session id", () => {
        initSession("test")
        resetSession()
        expect(getSession()).toBeNull()
        expect(getSessionId()).toBeNull()
    })

    it("saves state before clearing", () => {
        initSession("test")
        resetSession()
        expect(saveState).toHaveBeenCalledTimes(2) // once in init, once in reset
    })
})

describe("updateSession", () => {
    it("applies updater to current state", () => {
        initSession("test")
        updateSession((s) => {
            s.currentTurn = 10
        })
        expect(getSession()?.currentTurn).toBe(10)
    })

    it("saves after update", () => {
        initSession("test")
        vi.mocked(saveState).mockClear()
        updateSession((s) => {
            s.currentTurn = 5
        })
        expect(saveState).toHaveBeenCalled()
    })

    it("does nothing when no session", () => {
        updateSession((s) => {
            s.currentTurn = 999
        })
        expect(getSession()).toBeNull()
    })
})

describe("recordPrunedTokens", () => {
    it("increments prune counters", () => {
        initSession("test")
        recordPrunedTokens(100)
        recordPrunedTokens(50)
        const state = getSession()
        expect(state?.stats.pruneTokenCounter).toBe(150)
        expect(state?.stats.totalPruneTokens).toBe(150)
    })
})

describe("incrementTurn", () => {
    it("increments currentTurn", () => {
        initSession("test")
        incrementTurn()
        incrementTurn()
        expect(getSession()?.currentTurn).toBe(2)
    })
})
