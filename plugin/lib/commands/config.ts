import type { CommandContext } from "./index"
import { McpClient } from "../mcp-client"
import { appendDiagnostic } from "../logger"

export async function handleConfigCommand(ctx: CommandContext, mcpClient: McpClient): Promise<void> {
    const { section, key, value } = parseArgs(ctx.args)

    const lines = [
        "╔══════════════════════════════════════════╗",
        "║          Configuration                   ║",
        "╚══════════════════════════════════════════╝",
        "",
    ]

    if (section && key && value) {
        // Update config
        try {
            const result = await mcpClient.call("update_config", {
                section,
                key,
                value,
            })

            if (result?.status === "ok") {
                lines.push(`  Updated ${section}.${key} = ${value}`)
            } else {
                lines.push(`  Failed to update config`)
            }
        } catch (err) {
            lines.push(`  Error updating config: ${String(err)}`)
            appendDiagnostic(`config update failed: ${String(err)}`)
        }
    } else if (section) {
        // Show specific section
        try {
            const result = await mcpClient.call("update_config", {
                section,
                key: "_get",
                value: "",
            })
            lines.push(`  ${section}:`)
            lines.push(`  ${JSON.stringify(result?.value ?? {}, null, 2)}`)
        } catch (err) {
            lines.push(`  Error reading config: ${String(err)}`)
        }
    } else {
        // Show all config
        lines.push("  Config sections:")
        lines.push("    importance    - ML and heuristic scoring settings")
        lines.push("    pruning       - Context pruning thresholds")
        lines.push("    session       - Session recording options")
        lines.push("    languages     - Tree-sitter language settings")
        lines.push("    interconnectedness - File/function relationship graph")
        lines.push("")
        lines.push("  Usage:")
        lines.push("    /ctx config                       Show this help")
        lines.push("    /ctx config importance            Show importance settings")
        lines.push("    /ctx config importance threshold 5.0  Update a value")
    }

    ctx.client.tui.showToast(lines.join("\n"))
}

function parseArgs(args: string): { section: string; key: string; value: string } {
    const parts = args.trim().split(/\s+/)
    return {
        section: parts[0] ?? "",
        key: parts[1] ?? "",
        value: parts.slice(2).join(" "),
    }
}
