import type { CtxSessionState, ContextPriority } from "./types"
import { appendDiagnostic } from "../logger"

export function createSessionState(sessionId: string): CtxSessionState {
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

let currentState: CtxSessionState | null = null

export function getCurrentState(): CtxSessionState | null {
    return currentState
}

export function setCurrentState(state: CtxSessionState | null): void {
    currentState = state
}

export function buildPriorityMap(
    importanceScores: Record<string, number>,
    functionScores: Record<string, number>,
    blastRadius: Record<string, Array<{ target: string; weight: number }>>
): Map<string, ContextPriority> {
    const priority = new Map<string, ContextPriority>()

    for (const [file, score] of Object.entries(importanceScores)) {
        priority.set(file, {
            messageId: file,
            score,
            reason: `Importance score: ${score.toFixed(1)}`,
        })
    }

    for (const [func, score] of Object.entries(functionScores)) {
        const file = func.split("::")[0]
        const existing = priority.get(file)
        if (!existing || score > existing.score) {
            priority.set(file, {
                messageId: func,
                score,
                reason: `Function importance: ${score.toFixed(1)}`,
            })
        }
    }

    for (const [source, targets] of Object.entries(blastRadius)) {
        const sourcePriority = priority.get(source)
        if (!sourcePriority) continue

        for (const target of targets) {
            const targetPriority = priority.get(target.target)
            if (targetPriority) {
                const boosted = targetPriority.score * (0.5 + 0.5 * target.weight)
                if (boosted > targetPriority.score) {
                    priority.set(target.target, {
                        ...targetPriority,
                        score: boosted,
                        reason: `Blast radius from ${source} (weight: ${target.weight.toFixed(2)})`,
                    })
                }
            }
        }
    }

    return priority
}

export function scoreMessage(
    message: Record<string, unknown>,
    priorityMap: Map<string, ContextPriority>,
    protectedThreshold: number
): { score: number; preserved: boolean; reason: string } {
    const text = JSON.stringify(message)
    const filePattern = /(?:src|lib|app|tests?|packages?)\/[\w./-]+\.\w+/g
    const files = text.match(filePattern) ?? []

    let maxScore = 0
    let preserved = false
    const reasons: string[] = []

    for (const file of files) {
        const priority = priorityMap.get(file)
        if (priority) {
            if (priority.score >= protectedThreshold) {
                preserved = true
                reasons.push(`${file}: protected (${priority.score.toFixed(1)})`)
            }
            if (priority.score > maxScore) {
                maxScore = priority.score
            }
        }
    }

    return {
        score: maxScore,
        preserved,
        reason: preserved ? reasons.join(", ") : `Score: ${maxScore.toFixed(1)}`,
    }
}
