import type { CommandContext } from "./index"
import { McpClient } from "../mcp-client"
import { appendDiagnostic } from "../logger"

export async function handleFeedbackCommand(ctx: CommandContext, mcpClient: McpClient): Promise<void> {
    const lines = [
        "╔══════════════════════════════════════════╗",
        "║       Active Learning Feedback           ║",
        "╚══════════════════════════════════════════╝",
        "",
    ]

    try {
        const result = await mcpClient.call("get_pending_feedback", {})
        const samples = (result?.samples as Array<Record<string, unknown>>) ?? []

        if (samples.length === 0) {
            lines.push("  No pending samples for review.")
            lines.push("")
            lines.push("  Samples are generated after sessions when the")
            lines.push("  model is uncertain about file importance.")
        } else {
            lines.push(`  ${samples.length} sample(s) need your feedback:`)
            lines.push("")
            for (let i = 0; i < Math.min(samples.length, 5); i++) {
                const sample = samples[i]
                const filePath = (sample.file_path as string) ?? "unknown"
                const prob = (sample.probability as number[]) ?? [0.5, 0.5]
                const confidence = Math.max(prob[0], prob[1])

                lines.push(`  ${i + 1}. ${filePath}`)
                lines.push(`     Confidence: ${(confidence * 100).toFixed(1)}%`)
                lines.push(`     Model prediction: ${prob[1] > prob[0] ? "important" : "not important"}`)
                lines.push("")
            }

            if (samples.length > 5) {
                lines.push(`  ... and ${samples.length - 5} more`)
                lines.push("")
            }

            lines.push("  To label, use: /ctx feedback <number> <score>")
            lines.push("  Example: /ctx feedback 1 8.5")
        }
    } catch (err) {
        lines.push(`  Error: ${String(err)}`)
        appendDiagnostic(`feedback command failed: ${String(err)}`)
    }

    ctx.client.tui.showToast(lines.join("\n"))
}
