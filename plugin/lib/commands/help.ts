import type { CommandContext } from "./index"

const COMMANDS = [
    { name: "help", description: "Show this help message" },
    { name: "stats", description: "Show token usage, pruning stats, and importance distribution" },
    { name: "importance [file]", description: "Show importance scores for files/functions" },
    { name: "matrix [file]", description: "Show interconnectedness matrix for a file" },
    { name: "feedback", description: "Review and label active learning samples" },
    { name: "session", description: "Show current session data summary" },
    { name: "config", description: "Show or edit configuration" },
]

export function handleHelpCommand(ctx: CommandContext): void {
    const lines = [
        "╔══════════════════════════════════════════╗",
        "║       Context Manager Commands           ║",
        "╚══════════════════════════════════════════╝",
        "",
        ...COMMANDS.map((cmd) => `  /ctx ${cmd.name.padEnd(22)} ${cmd.description}`),
        "",
        "Example: /ctx importance src/auth/login.py",
        "Example: /ctx config importance.threshold 5.0",
    ]

    ctx.client.tui.showToast(lines.join("\n"))
}
