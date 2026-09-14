import * as fs from "node:fs"
import * as path from "node:path"
import type { SessionFile, SessionIndex, IndexEntry } from "./types"
import { appendDiagnostic } from "../logger"

export class SessionStorage {
    private sessionsDir: string
    private indexPath: string

    constructor(directory: string) {
        this.sessionsDir = path.join(directory, ".opencode", "sessions")
        this.indexPath = path.join(this.sessionsDir, "index.json")
    }

    async ensureDir(): Promise<void> {
        try {
            fs.mkdirSync(this.sessionsDir, { recursive: true })
        } catch (err) {
            appendDiagnostic(`Failed to create sessions dir: ${String(err)}`)
        }
    }

    async scanForCorruption(): Promise<void> {
        try {
            const files = fs.readdirSync(this.sessionsDir).filter((f) => f.endsWith(".json") && f !== "index.json")
            for (const file of files) {
                const filePath = path.join(this.sessionsDir, file)
                try {
                    const content = fs.readFileSync(filePath, "utf-8")
                    if (content.includes("\0")) {
                        appendDiagnostic(`Corruption detected in ${file}: NUL bytes found`)
                        const cleaned = content.replace(/\0+$/g, "").replace(/\0/g, "")
                        fs.writeFileSync(filePath, cleaned, "utf-8")
                    }
                } catch (err) {
                    appendDiagnostic(`Failed to read ${file}: ${String(err)}`)
                }
            }
        } catch (err) {
            appendDiagnostic(`scanForCorruption failed: ${String(err)}`)
        }
    }

    load(sessionId: string): SessionFile | null {
        const filePath = path.join(this.sessionsDir, `${sessionId}.json`)
        try {
            if (!fs.existsSync(filePath)) return null
            const content = fs.readFileSync(filePath, "utf-8")
            return JSON.parse(content) as SessionFile
        } catch (err) {
            appendDiagnostic(`Failed to load session ${sessionId}: ${String(err)}`)
            return null
        }
    }

    save(session: SessionFile): void {
        const filePath = path.join(this.sessionsDir, `${session._id}.json`)
        try {
            fs.writeFileSync(filePath, JSON.stringify(session, null, 2), "utf-8")
            this.updateIndex(session)
        } catch (err) {
            appendDiagnostic(`Failed to save session ${session._id}: ${String(err)}`)
        }
    }

    private updateIndex(session: SessionFile): void {
        try {
            let index: SessionIndex = { sessions: [] }
            if (fs.existsSync(this.indexPath)) {
                const content = fs.readFileSync(this.indexPath, "utf-8")
                index = JSON.parse(content) as SessionIndex
            }

            const existing = index.sessions.findIndex((s) => s.id === session._id)
            const entry: IndexEntry = {
                id: session._id,
                title: session.title,
                startedAt: session.startedAt,
                updatedAt: session.updatedAt,
                parentId: session.parentId,
            }

            if (existing >= 0) {
                index.sessions[existing] = entry
            } else {
                index.sessions.push(entry)
            }

            index.sessions.sort((a, b) => new Date(a.startedAt).getTime() - new Date(b.startedAt).getTime())
            fs.writeFileSync(this.indexPath, JSON.stringify(index, null, 2), "utf-8")
        } catch (err) {
            appendDiagnostic(`Failed to update index: ${String(err)}`)
        }
    }

    loadIndex(): SessionIndex {
        try {
            if (fs.existsSync(this.indexPath)) {
                const content = fs.readFileSync(this.indexPath, "utf-8")
                return JSON.parse(content) as SessionIndex
            }
        } catch (err) {
            appendDiagnostic(`Failed to load index: ${String(err)}`)
        }
        return { sessions: [] }
    }

    listSessionIds(): string[] {
        try {
            return fs
                .readdirSync(this.sessionsDir)
                .filter((f) => f.endsWith(".json") && f !== "index.json")
                .map((f) => f.replace(".json", ""))
        } catch {
            return []
        }
    }
}
