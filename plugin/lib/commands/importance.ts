import type { CommandContext } from "./index"
import { McpClient } from "../mcp-client"
import { appendDiagnostic } from "../logger"

export async function handleImportanceCommand(ctx: CommandContext, mcpClient: McpClient): Promise<void> {
    const { rest: target } = parseArgs(ctx.args)

    const lines = [
        "╔══════════════════════════════════════════╗",
        "║         Importance Scores                ║",
        "╚══════════════════════════════════════════╝",
        "",
    ]

    try {
        const result = await mcpClient.call("score_importance", {
            session_id: ctx.sessionId ?? "",
        })

        const files = (result?.files as Record<string, number>) ?? {}
        const functions = (result?.functions as Record<string, number>) ?? {}

        if (target) {
            // Show specific file/function
            const fileScore = files[target]
            if (fileScore !== undefined) {
                lines.push(`  File: ${target}`)
                lines.push(`  Score: ${fileScore.toFixed(2)} / 10.0`)
                lines.push(`  ${getBar(fileScore, 10)}`)
            }

            const funcEntries = Object.entries(functions).filter(([k]) => k.includes(target))
            if (funcEntries.length > 0) {
                lines.push("")
                lines.push("  Functions:")
                for (const [func, score] of funcEntries) {
                    lines.push(`    ${func}: ${score.toFixed(2)}`)
                }
            }

            if (fileScore === undefined && funcEntries.length === 0) {
                lines.push(`  No scores found for "${target}"`)
            }
        } else {
            // Show top files
            const sortedFiles = Object.entries(files)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 15)

            if (sortedFiles.length > 0) {
                lines.push("  Top Files by Importance:")
                for (const [file, score] of sortedFiles) {
                    lines.push(`    ${getBar(score, 10)} ${score.toFixed(1)}  ${file}`)
                }
            } else {
                lines.push("  No importance scores available.")
                lines.push("  Scores are computed during active sessions.")
            }
        }
    } catch (err) {
        lines.push(`  Error: ${String(err)}`)
        appendDiagnostic(`importance command failed: ${String(err)}`)
    }

    ctx.client.tui.showToast(lines.join("\n"))
}

function parseArgs(args: string): { rest: string } {
    return { rest: args.trim() }
}

function getBar(value: number, max: number): string {
    const width = 20
    const filled = Math.round((value / max) * width)
    return "█".repeat(Math.max(filled, 0)).padEnd(width, "░")
}
