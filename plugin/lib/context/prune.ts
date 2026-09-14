import type { CtxSessionState } from "./types"
import type { PluginConfig } from "../config"
import { buildPriorityMap, scoreMessage } from "./priority"
import { getSession } from "../state/state"
import { countTokens } from "../token-utils"
import { appendDiagnostic } from "../logger"

const PROTECTED_PLACEHOLDER = "[Content preserved — important file]"

export interface PruneResult {
    messages: Array<Record<string, unknown>>
    tokensRemoved: number
    filesPruned: string[]
    filesPreserved: string[]
    summaries: string[]
}

export function prune(
    messages: Array<Record<string, unknown>>,
    importanceScores: Record<string, number>,
    functionScores: Record<string, number>,
    blastRadius: Record<string, Array<{ target: string; weight: number }>>,
    config: PluginConfig
): PruneResult {
    const threshold = config.importance?.threshold ?? 3.0
    const protectedThreshold = config.importance?.protectedThreshold ?? 7.0
    const maxRatio = config.pruning?.maxCompressionRatio ?? 0.3

    const state = getSession()
    const priorityMap = buildPriorityMap(importanceScores, functionScores, blastRadius)

    const totalTokens = messages.reduce((sum, msg) => sum + countTokens(JSON.stringify(msg)), 0)
    const maxPrunable = Math.floor(totalTokens * maxRatio)
    let tokensRemoved = 0
    let prunedCount = 0

    const filesPruned: string[] = []
    const filesPreserved: string[] = []
    const summaries: string[] = []

    // Pass 1: Score all messages and identify tool outputs to prune
    const messageScores = messages.map((msg, idx) => {
        const result = scoreMessage(msg, priorityMap, protectedThreshold)
        return { idx, msg, ...result }
    })

    // Pass 2: Prune tool outputs for low-importance files
    const pruned = messages.map((msg, idx) => {
        const score = messageScores[idx]

        // Never prune protected messages
        if (score.preserved) {
            if (!filesPreserved.includes(score.reason)) {
                filesPreserved.push(score.reason)
            }
            return msg
        }

        // Check if this is a tool result with a prunable output
        if (isToolResult(msg) && score.score < threshold) {
            const toolOutput = getToolOutput(msg)
            if (toolOutput) {
                const outputTokens = countTokens(toolOutput)
                if (tokensRemoved + outputTokens <= maxPrunable) {
                    tokensRemoved += outputTokens
                    prunedCount++
                    const replaced = replaceToolOutput(msg, "[Output pruned — low importance file]")
                    if (!filesPruned.includes(score.reason)) {
                        filesPruned.push(score.reason)
                    }
                    return replaced
                }
            }
        }

        // Check if this is a regular message about a low-importance file
        if (score.score < threshold && !score.preserved) {
            const msgTokens = countTokens(JSON.stringify(msg))
            if (tokensRemoved + msgTokens <= maxPrunable) {
                tokensRemoved += msgTokens
                prunedCount++
                if (!filesPruned.includes(score.reason)) {
                    filesPruned.push(score.reason)
                }
                // Don't remove the message, just mark it as pruned for summary
                return msg
            }
        }

        return msg
    })

    // Pass 3: Build summary of what was pruned
    if (prunedCount > 0) {
        summaries.push(
            `Pruned ${prunedCount} messages, removed ~${tokensRemoved} tokens. ` +
            `Prunable files: ${filesPruned.join(", ") || "none"}. ` +
            `Protected: ${filesPreserved.length} files.`
        )
    }

    // Update state
    if (state) {
        state.stats.filesPruned = filesPruned.length
        state.stats.filesPreserved = filesPreserved.length
        state.stats.totalPruneTokens = tokensRemoved
    }

    return {
        messages: pruned,
        tokensRemoved,
        filesPruned,
        filesPreserved,
        summaries,
    }
}

function isToolResult(msg: Record<string, unknown>): boolean {
    const role = msg.role as string
    if (role !== "tool" && role !== "function") return false
    const content = msg.content ?? msg.parts
    if (!content) return false
    if (Array.isArray(content)) {
        return content.some((p: Record<string, unknown>) => p.type === "tool_result" || p.type === "tool")
    }
    return false
}

function getToolOutput(msg: Record<string, unknown>): string | null {
    const content = msg.content ?? msg.parts
    if (!content) return null
    if (typeof content === "string") return content
    if (Array.isArray(content)) {
        for (const part of content) {
            if (typeof part === "object" && part !== null) {
                const output = (part as Record<string, unknown>).output ??
                    (part as Record<string, unknown>).toolOutput ??
                    (part as Record<string, unknown>).text
                if (typeof output === "string") return output
            }
        }
    }
    return null
}

function replaceToolOutput(msg: Record<string, unknown>, replacement: string): Record<string, unknown> {
    const result = { ...msg }
    const content = msg.content ?? msg.parts
    if (Array.isArray(content)) {
        result.parts = content.map((part: Record<string, unknown>) => {
            if (part.type === "tool_result" || part.type === "tool") {
                return { ...part, output: replacement, toolOutput: replacement, text: replacement }
            }
            return part
        })
    } else if (typeof content === "string") {
        result.content = replacement
    }
    return result
}

export function getTokenBreakdown(
    messages: Array<Record<string, unknown>>,
    importanceScores: Record<string, number>,
    functionScores: Record<string, number>,
    blastRadius: Record<string, Array<{ target: string; weight: number }>>,
    config: PluginConfig
): {
    total: number
    prunable: number
    protected: number
    low: number
    mid: number
} {
    const protectedThreshold = config.importance?.protectedThreshold ?? 7.0
    const threshold = config.importance?.threshold ?? 3.0
    const priorityMap = buildPriorityMap(importanceScores, functionScores, blastRadius)

    let total = 0
    let prunable = 0
    let protectedCount = 0
    let low = 0
    let mid = 0

    for (const msg of messages) {
        const tokens = countTokens(JSON.stringify(msg))
        total += tokens
        const score = scoreMessage(msg, priorityMap, protectedThreshold)

        if (score.preserved) {
            protectedCount += tokens
        } else if (score.score < threshold) {
            prunable += tokens
            low += tokens
        } else {
            mid += tokens
        }
    }

    return { total, prunable, protected: protectedCount, low, mid }
}
