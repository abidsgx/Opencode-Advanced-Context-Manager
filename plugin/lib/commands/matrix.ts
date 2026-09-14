import type { CommandContext } from "./index"
import { McpClient } from "../mcp-client"
import { appendDiagnostic } from "../logger"

export async function handleMatrixCommand(ctx: CommandContext, mcpClient: McpClient): Promise<void> {
    const { rest: target } = parseArgs(ctx.args)

    const lines = [
        "╔══════════════════════════════════════════╗",
        "║      Interconnectedness Matrix           ║",
        "╚══════════════════════════════════════════╝",
        "",
    ]

    try {
        const result = await mcpClient.call("get_interconnectedness", {
            session_id: ctx.sessionId ?? "",
        })

        const files = (result?.files as Record<string, Record<string, { weight: number; relation: string }>>) ?? {}

        if (target) {
            // Show connections for specific file
            const connections = files[target]
            if (connections && Object.keys(connections).length > 0) {
                lines.push(`  Connections for ${target}:`)
                lines.push("")
                for (const [to, info] of Object.entries(connections)) {
                    const weightBar = "█".repeat(Math.round(info.weight * 10))
                    lines.push(`    → ${to}`)
                    lines.push(`      Relation: ${info.relation}`)
                    lines.push(`      Weight:   ${weightBar} ${info.weight.toFixed(2)}`)
                }
            } else {
                lines.push(`  No connections found for "${target}"`)
            }

            // Show reverse connections (who points to this file)
            const reverseConnections: Array<{ from: string; weight: number; relation: string }> = []
            for (const [from, targets] of Object.entries(files)) {
                if (targets[target]) {
                    reverseConnections.push({ from, ...targets[target] })
                }
            }

            if (reverseConnections.length > 0) {
                lines.push("")
                lines.push("  Referenced by:")
                for (const conn of reverseConnections) {
                    lines.push(`    ← ${conn.from} (${conn.relation}, weight: ${conn.weight.toFixed(2)})`)
                }
            }
        } else {
            // Show overview
            const fileCount = Object.keys(files).length
            let edgeCount = 0
            for (const targets of Object.values(files)) {
                edgeCount += Object.keys(targets).length
            }

            lines.push(`  Files in graph: ${fileCount}`)
            lines.push(`  Total edges: ${edgeCount}`)
            lines.push("")

            // Show most connected files
            const centrality: Record<string, number> = {}
            for (const [from, targets] of Object.entries(files)) {
                centrality[from] = (centrality[from] ?? 0) + Object.keys(targets).length
                for (const to of Object.keys(targets)) {
                    centrality[to] = (centrality[to] ?? 0) + 1
                }
            }

            const topFiles = Object.entries(centrality)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 10)

            if (topFiles.length > 0) {
                lines.push("  Most Connected Files:")
                for (const [file, count] of topFiles) {
                    lines.push(`    ${file}: ${count} connections`)
                }
            }
        }
    } catch (err) {
        lines.push(`  Error: ${String(err)}`)
        appendDiagnostic(`matrix command failed: ${String(err)}`)
    }

    ctx.client.tui.showToast(lines.join("\n"))
}

function parseArgs(args: string): { rest: string } {
    return { rest: args.trim() }
}
