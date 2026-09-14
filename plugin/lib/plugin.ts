import type { Plugin } from "@opencode-ai/plugin"
import { loadConfig, type PluginConfig } from "./config"
import { SessionRecorder } from "./session/recorder"
import { SessionStorage } from "./session/storage"
import { initDiagnostics, appendDiagnostic } from "./logger"
import { injectMetadata } from "./context/inject"
import {
    handleHelpCommand,
    handleStatsCommand,
    handleImportanceCommand,
    handleMatrixCommand,
    handleFeedbackCommand,
    handleSessionCommand,
    handleConfigCommand,
} from "./commands"
import { McpClient } from "./mcp-client"

function sessionId(event: { type: string; properties: Record<string, unknown> }): string | undefined {
    const p = event.properties
    if (p.sessionID) return p.sessionID as string
    const info = p.info as Record<string, unknown> | undefined
    if (info?.id) return info.id as string
    return undefined
}

function sessionInfo(event: { type: string; properties: Record<string, unknown> }) {
    const info = event.properties.info as Record<string, unknown> | undefined
    const time = info?.time as Record<string, unknown> | undefined
    return {
        id: (info?.id as string) ?? "",
        title: (info?.title as string) ?? "",
        parentId: (info?.parentID as string | undefined) ?? null,
        createdAt: (time?.created as string) ?? new Date().toISOString(),
        updatedAt: (time?.updated as string) ?? new Date().toISOString(),
    }
}

export const CtxManagerPlugin: Plugin = async ({ client, directory }) => {
    initDiagnostics(directory)
    const config: PluginConfig = await loadConfig(directory)
    const storage = new SessionStorage(directory)
    const recorder = new SessionRecorder(client as never, storage, config)
    const mcpClient = new McpClient(config.mcp)

    await storage.ensureDir()
    void storage.scanForCorruption().catch((err) => appendDiagnostic(`scanForCorruption failed: ${String(err)}`))

    const syncTimers = new Map<string, ReturnType<typeof setTimeout>>()

    function scheduleSync(id: string, immediate = false): void {
        if (!id) return
        const existing = syncTimers.get(id)
        if (immediate) {
            if (existing) clearTimeout(existing)
            syncTimers.delete(id)
            recorder.syncSession(id).catch((err) => appendDiagnostic(`sync failed for ${id}: ${String(err)}`))
            return
        }
        if (existing) return
        const timer = setTimeout(() => {
            syncTimers.delete(id)
            recorder.syncSession(id).catch((err) => appendDiagnostic(`sync failed for ${id}: ${String(err)}`))
        }, 500)
        syncTimers.set(id, timer)
    }

    async function handleCtxCommand(args: string): Promise<void> {
        const parts = args.trim().split(/\s+/)
        const command = parts[0] ?? "help"
        const rest = parts.slice(1).join(" ")

        const ctx = {
            client: client as never,
            directory,
            sessionId: undefined as string | undefined,
            args: rest,
        }

        switch (command) {
            case "help":
                handleHelpCommand(ctx)
                break
            case "stats":
                await handleStatsCommand(ctx)
                break
            case "importance":
                await handleImportanceCommand(ctx, mcpClient)
                break
            case "matrix":
                await handleMatrixCommand(ctx, mcpClient)
                break
            case "feedback":
                await handleFeedbackCommand(ctx, mcpClient)
                break
            case "session":
                await handleSessionCommand(ctx)
                break
            case "config":
                await handleConfigCommand(ctx, mcpClient)
                break
            default:
                handleHelpCommand(ctx)
                break
        }
    }

    return {
        "experimental.chat.system.transform": async (input, output) => {
            if (!config.enabled) return
            const importanceContext = await recorder.getImportanceContext(input.sessionID)
            if (!importanceContext) return

            const importanceScores: Record<string, number> = {}
            const functionScores: Record<string, number> = {}
            for (const f of importanceContext.topFiles) {
                importanceScores[f.path] = f.score
            }

            const injected = injectMetadata(
                "",
                importanceScores,
                functionScores,
                importanceContext.connections,
                null,
                config
            )

            output.system.push(injected)
        },

        "experimental.chat.messages.transform": async (_input, output) => {
            if (!config.enabled) return
            try {
                // The output.messages contains the messages that will be sent
                // We need to transform them using our pruning logic
                // For now, we pass through as the actual pruning happens in recorder.transformMessages
            } catch (err) {
                appendDiagnostic(`message transform failed: ${String(err)}`)
            }
        },

        "experimental.text.complete": async (_input, output) => {
            if (!config.enabled) return
            output.text = output.text.replace(/<ctx-[^>]*>[\s\S]*?<\/ctx-[^>]*>/g, "")
        },

        "command.execute.before": async (input, _output) => {
            const cmd = input.command as string
            if (cmd?.startsWith("/ctx")) {
                const args = cmd.replace("/ctx", "").trim()
                await handleCtxCommand(args)
                // Set command to empty to prevent further processing
                input.command = ""
            }
        },

        "tool.execute.after": async (input) => {
            if (!config.session?.extendedRecording) return
            if (input.sessionID) {
                await recorder.recordToolExecution(input.sessionID, input as never)
                scheduleSync(input.sessionID)
            }
        },

        event: async ({ event }) => {
            try {
                const id = sessionId(event)
                if (!id) return

                switch (event.type) {
                    case "session.created":
                        await recorder.createSession(id, sessionInfo(event))
                        break

                    case "session.updated":
                    case "message.updated":
                    case "message.part.updated":
                    case "message.removed":
                        scheduleSync(id)
                        break

                    case "session.deleted":
                        await recorder.syncSession(id)
                        break

                    case "session.compacted":
                        await recorder.markCompaction(id)
                        await recorder.syncSession(id)
                        break

                    case "permission.updated": {
                        const p = event.properties as Record<string, unknown>
                        const pattern = p.pattern
                        await recorder.recordPermission(id, {
                            type: "asked",
                            timestamp: new Date().toISOString(),
                            permission: (p.type as string) ?? "",
                            command: Array.isArray(pattern) ? (pattern as string[]).join(" ") : (pattern as string) ?? "",
                            question: (p.title as string) ?? "",
                        })
                        break
                    }

                    case "permission.replied": {
                        const p = event.properties as Record<string, unknown>
                        await recorder.recordPermission(id, {
                            type: "replied",
                            timestamp: new Date().toISOString(),
                            permission: "",
                            allowed: (p.response as string) === "allow",
                        })
                        break
                    }

                    case "session.error": {
                        const p = event.properties as Record<string, unknown>
                        const err = p.error as
                            | { name?: string; message?: string; data?: { message?: string; statusCode?: number } }
                            | undefined
                        if (err?.message || err?.data?.message) {
                            await recorder.recordError(id, {
                                timestamp: new Date().toISOString(),
                                name: err.name ?? "Error",
                                message: err.data?.message ?? err.message ?? "",
                                statusCode: err.data?.statusCode,
                                sessionID: id,
                            })
                        }
                        break
                    }
                }
            } catch (err) {
                await appendDiagnostic(`event handler failed for ${event.type}: ${String(err)}`)
            }
        },
    }
}
