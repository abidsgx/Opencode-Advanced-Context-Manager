import type { PluginConfig } from "../config"
import type { SessionStorage } from "./storage"
import type { SessionFile, Turn, Part, PermissionEvent, SessionError, SessionInfo } from "./types"
import { appendDiagnostic } from "../logger"
import { countTokens } from "../token-utils"
import { McpClient } from "../mcp-client"
import { prune, getTokenBreakdown } from "../context/prune"
import { buildPriorityMap } from "../context/priority"
import { initSession, getSession } from "../state/state"

interface ApiClient {
    session: {
        get: (opts: { path: { id: string } }) => Promise<{ data?: Record<string, unknown> }>
        messages: (opts: { path: { id: string } }) => Promise<{ data?: Array<Record<string, unknown>> }>
    }
}

function toPart(raw: Record<string, unknown>): Part {
    const type = (raw.type as string) ?? "text"
    return {
        type: type as Part["type"],
        text: raw.text as string | undefined,
        toolName: raw.toolName as string | undefined,
        toolInput: raw.toolInput as Record<string, unknown> | undefined,
        toolOutput: raw.toolOutput as string | undefined,
        isError: raw.isError as boolean | undefined,
        fileName: raw.fileName as string | undefined,
    }
}

function toTurn(raw: Record<string, unknown>): Turn {
    const parts = ((raw.parts as Array<Record<string, unknown>>) ?? []).map(toPart)
    const error = raw.error as { name?: string; message?: string; statusCode?: number } | undefined

    return {
        role: (raw.role as "user" | "assistant") ?? "user",
        messageId: (raw.messageId as string) ?? (raw.id as string) ?? "",
        timestamp: (raw.timestamp as string) ?? new Date().toISOString(),
        parts,
        error: error
            ? {
                  name: error.name ?? "Error",
                  message: error.message ?? "",
                  statusCode: error.statusCode,
              }
            : undefined,
    }
}

export class SessionRecorder {
    private client: ApiClient
    private storage: SessionStorage
    private config: PluginConfig
    private mcpClient: McpClient | null = null
    private knownIds: Map<string, Set<string>> = new Map()
    private syncLocks: Map<string, Promise<void>> = new Map()

    constructor(client: ApiClient, storage: SessionStorage, config: PluginConfig) {
        this.client = client
        this.storage = storage
        this.config = config
    }

    private getMcpClient(): McpClient {
        if (!this.mcpClient) {
            this.mcpClient = new McpClient(this.config.mcp)
        }
        return this.mcpClient
    }

    async createSession(id: string, info: SessionInfo): Promise<void> {
        const now = new Date().toISOString()
        const session: SessionFile = {
            _id: id,
            title: info.title,
            startedAt: now,
            updatedAt: now,
            compactedAt: null,
            parentId: info.parentId,
            subSessions: [],
            goal: null,
            turns: [],
            permissions: [],
            errors: [],
            editImpact: [],
            importanceScores: null,
            interconnectedness: null,
        }

        if (info.parentId) {
            const parent = this.storage.load(info.parentId)
            if (parent) {
                parent.subSessions.push(id)
                this.storage.save(parent)
            }
        }

        this.storage.save(session)
        this.knownIds.set(id, new Set())
    }

    async syncSession(id: string): Promise<void> {
        const existing = this.locks().get(id)
        if (existing) {
            await existing
            return
        }

        const lock = this.doSync(id)
        this.locks().set(id, lock)
        try {
            await lock
        } finally {
            this.locks().delete(id)
        }
    }

    private locks(): Map<string, Promise<void>> {
        return this.syncLocks
    }

    private async doSync(id: string): Promise<void> {
        let session = this.storage.load(id)
        if (!session) {
            session = await this.rebuildSession(id)
            if (!session) return
        }

        try {
            const response = await this.client.session.messages({ path: { id } })
            const messages = (response.data as Array<Record<string, unknown>>) ?? []

            if (!this.knownIds.has(id)) {
                this.knownIds.set(id, new Set(session.turns.map((t) => t.messageId)))
            }
            const known = this.knownIds.get(id)!

            const newTurns: Turn[] = []
            for (const msg of messages) {
                const msgId = (msg.messageId as string) ?? (msg.id as string) ?? ""
                if (!known.has(msgId)) {
                    known.add(msgId)
                    newTurns.push(toTurn(msg))
                }
            }

            if (newTurns.length > 0) {
                session.turns.push(...newTurns)
                session.updatedAt = new Date().toISOString()

                if (this.config.session?.captureEditMetadata) {
                    await this.enrichEditMetadata(session, newTurns)
                }

                this.storage.save(session)
            }
        } catch (err) {
            appendDiagnostic(`syncSession API call failed for ${id}: ${String(err)}`)
        }
    }

    private async enrichEditMetadata(session: SessionFile, turns: Turn[]): Promise<void> {
        for (const turn of turns) {
            for (const part of turn.parts) {
                if (part.type === "tool" && part.toolName?.includes("edit") && part.toolInput) {
                    const filePath = (part.toolInput.path as string) ?? (part.toolInput.file as string) ?? null
                    if (filePath && !part.editMetadata) {
                        part.editMetadata = {
                            reason: "",
                            contributesToGoal: "",
                            success: !part.isError,
                            tradeoffs: undefined,
                        }
                    }
                }
            }
        }
    }

    async recordToolExecution(sessionId: string, input: Record<string, unknown>): Promise<void> {
        const session = this.storage.load(sessionId)
        if (!session) return

        const toolName = input.toolName as string | undefined
        const toolInput = input.input as Record<string, unknown> | undefined
        const toolOutput = input.output as string | undefined
        const isError = input.isError as boolean | undefined

        if (toolName?.includes("edit") && toolInput) {
            const filePath = (toolInput.path as string) ?? (toolInput.file as string) ?? null
            if (filePath) {
                session.editImpact.push({
                    file: filePath,
                    function: null,
                    timestamp: new Date().toISOString(),
                    change: toolOutput ?? "",
                    reason: "",
                    impact: "",
                    tested: false,
                })
            }
        }

        session.updatedAt = new Date().toISOString()
        this.storage.save(session)
    }

    async recordPermission(sessionId: string, permission: PermissionEvent): Promise<void> {
        const session = this.storage.load(sessionId)
        if (!session) return

        session.permissions.push(permission)
        session.updatedAt = new Date().toISOString()
        this.storage.save(session)
    }

    async recordError(sessionId: string, error: SessionError): Promise<void> {
        const session = this.storage.load(sessionId)
        if (!session) return

        session.errors.push(error)
        session.updatedAt = new Date().toISOString()
        this.storage.save(session)
    }

    async markCompaction(sessionId: string): Promise<void> {
        const session = this.storage.load(sessionId)
        if (!session) return

        session.compactedAt = new Date().toISOString()
        session.updatedAt = new Date().toISOString()
        this.storage.save(session)
    }

    async getImportanceContext(sessionId: string | undefined): Promise<{
        topFiles: Array<{ path: string; score: number }>
        connections: Array<{ from: string; to: string; weight: number; relation: string }>
    } | null> {
        if (!sessionId) return null

        try {
            const mcp = this.getMcpClient()
            const importance = await mcp.call("get_importance", { session_id: sessionId })
            const graph = await mcp.call("get_interconnectedness", { session_id: sessionId })

            const topFiles = Object.entries((importance?.files as Record<string, number>) ?? {})
                .sort(([, a], [, b]) => b - a)
                .slice(0, 10)
                .map(([path, score]) => ({ path, score }))

            const connections: Array<{ from: string; to: string; weight: number; relation: string }> = []
            const fileEdges = (importance?.files as Record<string, Record<string, { weight: number; relation: string }>>) ?? {}
            for (const [from, targets] of Object.entries(fileEdges)) {
                for (const [to, info] of Object.entries(targets)) {
                    connections.push({ from, to, weight: info.weight, relation: info.relation })
                }
            }

            return { topFiles, connections: connections.slice(0, 15) }
        } catch (err) {
            appendDiagnostic(`getImportanceContext failed: ${String(err)}`)
            return null
        }
    }

    async transformMessages(input: {
        messages: Array<Record<string, unknown>>
        sessionID?: string
    }): Promise<{ messages: Array<Record<string, unknown>> }> {
        if (!this.config.pruning?.enabled) return input
        if (!input.sessionID) return input

        // Initialize state for this session
        const state = initSession(input.sessionID)

        let importanceScores: Record<string, number> = {}
        let functionScores: Record<string, number> = {}
        let connections: Array<{ from: string; to: string; weight: number; relation: string }> = []

        try {
            const mcp = this.getMcpClient()
            const importanceResult = await mcp.call("score_importance", {
                session_id: input.sessionID,
            })
            importanceScores = (importanceResult?.files as Record<string, number>) ?? {}
            functionScores = (importanceResult?.functions as Record<string, number>) ?? {}

            const graphResult = await mcp.call("get_interconnectedness", {
                session_id: input.sessionID,
            })
            const fileEdges = (graphResult?.files as Record<string, Record<string, { weight: number; relation: string }>>) ?? {}
            for (const [from, targets] of Object.entries(fileEdges)) {
                for (const [to, info] of Object.entries(targets)) {
                    connections.push({ from, to, weight: info.weight, relation: info.relation })
                }
            }
        } catch (err) {
            appendDiagnostic(`transformMessages MCP call failed: ${String(err)}`)
            // Fall back to no pruning
            return input
        }

        // Build blast radius from connections
        const blastRadius: Record<string, Array<{ target: string; weight: number }>> = {}
        for (const conn of connections) {
            if (!blastRadius[conn.from]) blastRadius[conn.from] = []
            blastRadius[conn.from].push({ target: conn.to, weight: conn.weight })
        }

        // Run importance-based pruning
        const result = prune(
            input.messages,
            importanceScores,
            functionScores,
            blastRadius,
            this.config
        )

        // Log pruning stats
        if (result.summaries.length > 0) {
            for (const summary of result.summaries) {
                appendDiagnostic(summary)
            }
        }

        return { messages: result.messages }
    }

    private async rebuildSession(id: string): Promise<SessionFile | null> {
        try {
            const response = await this.client.session.get({ path: { id } })
            const data = response.data as Record<string, unknown> | undefined
            if (!data) return null

            const now = new Date().toISOString()
            const session: SessionFile = {
                _id: id,
                title: (data.title as string) ?? "",
                startedAt: (data.createdAt as string) ?? now,
                updatedAt: now,
                compactedAt: null,
                parentId: (data.parentId as string) ?? null,
                subSessions: [],
                goal: null,
                turns: [],
                permissions: [],
                errors: [],
                editImpact: [],
                importanceScores: null,
                interconnectedness: null,
            }

            this.storage.save(session)
            return session
        } catch (err) {
            appendDiagnostic(`rebuildSession failed for ${id}: ${String(err)}`)
            return null
        }
    }
}
