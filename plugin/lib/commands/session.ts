import type { CommandContext } from "./index"
import { SessionStorage } from "../session/storage"
import { appendDiagnostic } from "../logger"

export async function handleSessionCommand(ctx: CommandContext): Promise<void> {
    const lines = [
        "╔══════════════════════════════════════════╗",
        "║          Session Summary                 ║",
        "╚══════════════════════════════════════════╝",
        "",
    ]

    if (!ctx.sessionId) {
        lines.push("  No active session.")
        ctx.client.tui.showToast(lines.join("\n"))
        return
    }

    try {
        const storage = new SessionStorage(ctx.directory)
        const session = storage.load(ctx.sessionId)

        if (!session) {
            lines.push(`  Session not found: ${ctx.sessionId}`)
            ctx.client.tui.showToast(lines.join("\n"))
            return
        }

        lines.push(`  ID: ${session._id}`)
        lines.push(`  Title: ${session.title || "(untitled)"}`)
        lines.push(`  Started: ${session.startedAt}`)
        lines.push(`  Updated: ${session.updatedAt}`)
        lines.push(`  Goal: ${session.goal ?? "(none)"}`)
        lines.push("")
        lines.push("  ── Activity ──")
        lines.push(`  Turns: ${session.turns.length}`)
        lines.push(`  Edits: ${session.editImpact.length}`)
        lines.push(`  Permissions: ${session.permissions.length}`)
        lines.push(`  Errors: ${session.errors.length}`)

        if (session.editImpact.length > 0) {
            lines.push("")
            lines.push("  ── Files Edited ──")
            const files = [...new Set(session.editImpact.map((e) => e.file))]
            for (const file of files.slice(0, 10)) {
                const edits = session.editImpact.filter((e) => e.file === file)
                lines.push(`    ${file} (${edits.length} edit${edits.length > 1 ? "s" : ""})`)
            }
            if (files.length > 10) {
                lines.push(`    ... and ${files.length - 10} more files`)
            }
        }

        if (session.importanceScores) {
            lines.push("")
            lines.push("  ── Importance Scores ──")
            const topFiles = Object.entries(session.importanceScores.files ?? {})
                .sort(([, a], [, b]) => b - a)
                .slice(0, 5)
            for (const [file, score] of topFiles) {
                lines.push(`    ${file}: ${score.toFixed(1)}`)
            }
        }
    } catch (err) {
        lines.push(`  Error: ${String(err)}`)
        appendDiagnostic(`session command failed: ${String(err)}`)
    }

    ctx.client.tui.showToast(lines.join("\n"))
}
