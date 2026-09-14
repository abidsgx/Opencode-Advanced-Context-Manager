export { handleHelpCommand } from "./help"
export { handleStatsCommand } from "./stats"
export { handleImportanceCommand } from "./importance"
export { handleMatrixCommand } from "./matrix"
export { handleFeedbackCommand } from "./feedback"
export { handleSessionCommand } from "./session"
export { handleConfigCommand } from "./config"

export interface CommandContext {
    client: {
        tui: {
            showToast: (msg: string) => void
            showChat: (msg: string) => void
        }
    }
    directory: string
    sessionId?: string
    args: string
}

export function parseArgs(args: string): { command: string; rest: string } {
    const parts = args.trim().split(/\s+/)
    const command = parts[0] ?? ""
    const rest = parts.slice(1).join(" ")
    return { command, rest }
}
