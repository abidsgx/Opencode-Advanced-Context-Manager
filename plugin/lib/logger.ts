import * as fs from "node:fs"
import * as path from "node:path"

let logDir: string | null = null

export function initDiagnostics(directory: string): void {
    logDir = path.join(directory, ".opencode", "logs", "ctx-manager")
    try {
        fs.mkdirSync(logDir, { recursive: true })
    } catch {
        logDir = null
    }
}

export function appendDiagnostic(message: string): void {
    if (!logDir) return
    try {
        const timestamp = new Date().toISOString()
        const logFile = path.join(logDir, `${new Date().toISOString().split("T")[0]}.log`)
        fs.appendFileSync(logFile, `[${timestamp}] ${message}\n`)
    } catch {
        // Never throw from logging
    }
}
