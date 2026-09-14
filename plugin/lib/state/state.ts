import type { CtxSessionState } from "../context/types"
import { createSessionState } from "../context/priority"
import { loadState, saveState } from "./persistence"
import { appendDiagnostic } from "../logger"

let currentState: CtxSessionState | null = null
let currentSessionId: string | null = null

export function initSession(sessionId: string): CtxSessionState {
    if (currentSessionId === sessionId && currentState) {
        return currentState
    }

    const loaded = loadState(sessionId)
    if (loaded) {
        currentState = loaded
        currentSessionId = sessionId
        return currentState
    }

    currentState = createSessionState(sessionId)
    currentSessionId = sessionId
    saveState(currentState)
    return currentState
}

export function getSession(): CtxSessionState | null {
    return currentState
}

export function getSessionId(): string | null {
    return currentSessionId
}

export function resetSession(): void {
    if (currentState && currentSessionId) {
        saveState(currentState)
    }
    currentState = null
    currentSessionId = null
}

export function updateSession(updater: (state: CtxSessionState) => void): void {
    if (!currentState) return
    updater(currentState)
    saveState(currentState)
}

export function recordPrunedTokens(tokenCount: number): void {
    if (!currentState) return
    currentState.stats.pruneTokenCounter += tokenCount
    currentState.stats.totalPruneTokens += tokenCount
    saveState(currentState)
}

export function incrementTurn(): void {
    if (!currentState) return
    currentState.currentTurn++
    saveState(currentState)
}
