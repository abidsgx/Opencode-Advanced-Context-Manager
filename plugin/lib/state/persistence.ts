import * as fs from "node:fs"
import * as path from "node:path"
import type { CtxSessionState } from "../context/types"
import { appendDiagnostic } from "../logger"

const STATE_DIR = "ctx-manager/state"

function getStateDir(directory: string): string {
    return path.join(directory, ".opencode", STATE_DIR)
}

function getStatePath(directory: string, sessionId: string): string {
    return path.join(getStateDir(directory), `${sessionId}.json`)
}

function getDirectory(): string {
    return process.cwd()
}

function serializeState(state: CtxSessionState): Record<string, unknown> {
    return {
        sessionId: state.sessionId,
        currentTurn: state.currentTurn,
        stats: state.stats,
        modelContextLimit: state.modelContextLimit,
        systemPromptTokens: state.systemPromptTokens,
        priority: Object.fromEntries(state.priority.entries()),
        compressionBlocks: Object.fromEntries(
            Array.from(state.compressionBlocks.entries()).map(([k, v]) => [k, v])
        ),
        prunedMessages: Object.fromEntries(
            Array.from(state.prunedMessages.entries()).map(([k, v]) => [k, v])
        ),
        messageIds: {
            byRawId: Object.fromEntries(state.messageIds.byRawId.entries()),
            byRef: Object.fromEntries(state.messageIds.byRef.entries()),
            nextRef: state.messageIds.nextRef,
        },
    }
}

function deserializeState(data: Record<string, unknown>): CtxSessionState {
    const state = createSessionState(data.sessionId as string)
    state.currentTurn = (data.currentTurn as number) ?? 0
    state.stats = data.stats as CtxSessionState["stats"] ?? state.stats
    state.modelContextLimit = data.modelContextLimit as number | undefined
    state.systemPromptTokens = data.systemPromptTokens as number | undefined

    if (data.priority && typeof data.priority === "object") {
        for (const [k, v] of Object.entries(data.priority as Record<string, unknown>)) {
            state.priority.set(k, v as never)
        }
    }

    if (data.compressionBlocks && typeof data.compressionBlocks === "object") {
        for (const [k, v] of Object.entries(data.compressionBlocks as Record<string, unknown>)) {
            state.compressionBlocks.set(Number(k), v as never)
        }
    }

    if (data.prunedMessages && typeof data.prunedMessages === "object") {
        for (const [k, v] of Object.entries(data.prunedMessages as Record<string, unknown>)) {
            state.prunedMessages.set(k, v as never)
        }
    }

    if (data.messageIds && typeof data.messageIds === "object") {
        const msgIds = data.messageIds as Record<string, unknown>
        if (msgIds.byRawId && typeof msgIds.byRawId === "object") {
            for (const [k, v] of Object.entries(msgIds.byRawId as Record<string, string>)) {
                state.messageIds.byRawId.set(k, v)
            }
        }
        if (msgIds.byRef && typeof msgIds.byRef === "object") {
            for (const [k, v] of Object.entries(msgIds.byRef as Record<string, string>)) {
                state.messageIds.byRef.set(k, v)
            }
        }
        state.messageIds.nextRef = (msgIds.nextRef as number) ?? 1
    }

    return state
}

export function loadState(sessionId: string): CtxSessionState | null {
    const dir = getDirectory()
    const filePath = getStatePath(dir, sessionId)
    try {
        if (!fs.existsSync(filePath)) return null
        const content = fs.readFileSync(filePath, "utf-8")
        const data = JSON.parse(content) as Record<string, unknown>
        return deserializeState(data)
    } catch (err) {
        appendDiagnostic(`Failed to load state for ${sessionId}: ${String(err)}`)
        return null
    }
}

export function saveState(state: CtxSessionState): void {
    const dir = getDirectory()
    const stateDir = getStateDir(dir)
    try {
        fs.mkdirSync(stateDir, { recursive: true })
        const filePath = getStatePath(dir, state.sessionId ?? "unknown")
        const data = serializeState(state)
        fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf-8")
    } catch (err) {
        appendDiagnostic(`Failed to save state: ${String(err)}`)
    }
}

function createSessionState(sessionId: string): CtxSessionState {
    return {
        sessionId,
        currentTurn: 0,
        priority: new Map(),
        compressionBlocks: new Map(),
        prunedMessages: new Map(),
        messageIds: {
            byRawId: new Map(),
            byRef: new Map(),
            nextRef: 1,
        },
        stats: {
            pruneTokenCounter: 0,
            totalPruneTokens: 0,
            filesPruned: 0,
            filesPreserved: 0,
        },
        modelContextLimit: undefined,
        systemPromptTokens: undefined,
    }
}
