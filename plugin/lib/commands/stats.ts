import type { CommandContext } from "./index"
import { getSession } from "../state/state"
import { countTokens } from "../token-utils"
import { appendDiagnostic } from "../logger"

export async function handleStatsCommand(ctx: CommandContext): Promise<void> {
    const state = getSession()

    const lines = [
        "╔══════════════════════════════════════════╗",
        "║          Context Manager Stats           ║",
        "╚══════════════════════════════════════════╝",
        "",
    ]

    if (!state) {
        lines.push("  No active session.")
        ctx.client.tui.showToast(lines.join("\n"))
        return
    }

    const stats = state.stats
    lines.push(`  Session: ${state.sessionId ?? "unknown"}`)
    lines.push(`  Current Turn: ${state.currentTurn}`)
    lines.push("")
    lines.push("  ── Pruning Stats ──")
    lines.push(`  Total tokens pruned: ${stats.totalPruneTokens}`)
    lines.push(`  Files pruned: ${stats.filesPruned}`)
    lines.push(`  Files preserved: ${stats.filesPreserved}`)
    lines.push("")

    if (state.modelContextLimit) {
        lines.push(`  Model context limit: ${state.modelContextLimit.toLocaleString()} tokens`)
    }
    if (state.systemPromptTokens) {
        lines.push(`  System prompt tokens: ${state.systemPromptTokens.toLocaleString()}`)
    }

    lines.push("")
    lines.push(`  Priority entries: ${state.priority.size}`)
    lines.push(`  Compression blocks: ${state.compressionBlocks.size}`)
    lines.push(`  Pruned messages: ${state.prunedMessages.size}`)

    ctx.client.tui.showToast(lines.join("\n"))
}
