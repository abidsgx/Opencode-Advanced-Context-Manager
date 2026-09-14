import type { PluginConfig } from "../config"

export interface ImportanceContext {
    topFiles: Array<{ path: string; score: number }>
    connections: Array<{ from: string; to: string; weight: number; relation: string }>
    goal: string | null
}

export function buildSystemPromptExtension(
    importanceScores: Record<string, number>,
    functionScores: Record<string, number>,
    connections: Array<{ from: string; to: string; weight: number; relation: string }>,
    goal: string | null
): string {
    const topFiles = Object.entries(importanceScores)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10)
        .map(([path, score]) => `  - ${path}: ${score.toFixed(1)}`)

    const topFunctions = Object.entries(functionScores)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10)
        .map(([func, score]) => `  - ${func}: ${score.toFixed(1)}`)

    const connectionHints = connections
        .filter((c) => c.weight > 0.5)
        .slice(0, 15)
        .map((c) => `  - ${c.from} → ${c.to} (${c.relation}, weight: ${c.weight.toFixed(2)})`)

    const parts = ["\n\n<ctx-manager>"]

    if (goal) {
        parts.push(`## Current Goal: ${goal}`)
    }

    if (topFiles.length > 0) {
        parts.push("## File Importance Scores (0-10)")
        parts.push(topFiles.join("\n"))
    }

    if (topFunctions.length > 0) {
        parts.push("## Function Importance Scores (0-10)")
        parts.push(topFunctions.join("\n"))
    }

    if (connectionHints.length > 0) {
        parts.push("## Key Interconnections")
        parts.push(connectionHints.join("\n"))
        parts.push("Note: Changes to these files may affect connected files.")
    }

    parts.push("</ctx-manager>")

    return parts.join("\n")
}

export function buildPruningNotification(
    filesPruned: string[],
    filesPreserved: string[],
    tokensRemoved: number
): string {
    const parts = ["<ctx-manager-notification>"]

    if (filesPruned.length > 0) {
        parts.push(`Pruned ${filesPruned.length} low-importance file references (~${tokensRemoved} tokens).`)
    }

    if (filesPreserved.length > 0) {
        parts.push(`Preserved ${filesPreserved.length} high-importance files.`)
    }

    parts.push("</ctx-manager-notification>")
    return parts.join("\n")
}

export function injectMetadata(
    systemPrompt: string,
    importanceScores: Record<string, number>,
    functionScores: Record<string, number>,
    connections: Array<{ from: string; to: string; weight: number; relation: string }>,
    goal: string | null,
    config: PluginConfig
): string {
    if (!config.enabled) return systemPrompt

    const extension = buildSystemPromptExtension(
        importanceScores,
        functionScores,
        connections,
        goal
    )

    return systemPrompt + extension
}
