import { spawn, type ChildProcess } from "node:child_process"
import type { McpConfig } from "./config"
import { appendDiagnostic } from "./logger"

export interface McpToolResult {
    content: Array<{ type: string; text: string }>
}

export class McpClient {
    private process: ChildProcess | null = null
    private config: McpConfig
    private requestId: number = 0
    private pending: Map<number, { resolve: (result: unknown) => void; reject: (error: Error) => void }> = new Map()
    private buffer: string = ""

    constructor(config: McpConfig) {
        this.config = config
    }

    private ensureProcess(): ChildProcess {
        if (this.process && !this.process.killed) {
            return this.process
        }

        this.process = spawn(this.config.command, this.config.args, {
            cwd: this.config.cwd,
            stdio: ["pipe", "pipe", "pipe"],
        })

        this.process.stdout?.on("data", (data: Buffer) => {
            this.buffer += data.toString()
            this.processMessages()
        })

        this.process.stderr?.on("data", (data: Buffer) => {
            appendDiagnostic(`MCP server stderr: ${data.toString()}`)
        })

        this.process.on("error", (err) => {
            appendDiagnostic(`MCP process error: ${String(err)}`)
        })

        this.process.on("close", () => {
            this.process = null
            for (const [, pending] of this.pending) {
                pending.reject(new Error("MCP process closed"))
            }
            this.pending.clear()
        })

        return this.process
    }

    private processMessages(): void {
        const lines = this.buffer.split("\n")
        this.buffer = lines.pop() ?? ""

        for (const line of lines) {
            if (!line.trim()) continue
            try {
                const message = JSON.parse(line)
                if (message.id !== undefined && this.pending.has(message.id)) {
                    const pending = this.pending.get(message.id)!
                    this.pending.delete(message.id)
                    if (message.error) {
                        pending.reject(new Error(message.error.message ?? "Unknown MCP error"))
                    } else {
                        pending.resolve(message.result)
                    }
                }
            } catch {
                // Ignore non-JSON lines
            }
        }
    }

    async call(tool: string, args: Record<string, unknown>): Promise<Record<string, unknown> | null> {
        try {
            const proc = this.ensureProcess()
            const id = ++this.requestId

            const request = {
                jsonrpc: "2.0",
                id,
                method: "tools/call",
                params: {
                    name: tool,
                    arguments: args,
                },
            }

            return new Promise((resolve, reject) => {
                this.pending.set(id, {
                    resolve: (result: unknown) => {
                        const mcpResult = result as McpToolResult
                        if (mcpResult?.content?.[0]?.text) {
                            try {
                                resolve(JSON.parse(mcpResult.content[0].text))
                            } catch {
                                resolve({ text: mcpResult.content[0].text })
                            }
                        } else {
                            resolve(result as Record<string, unknown> | null)
                        }
                    },
                    reject,
                })

                proc.stdin?.write(JSON.stringify(request) + "\n")

                setTimeout(() => {
                    if (this.pending.has(id)) {
                        this.pending.delete(id)
                        reject(new Error(`MCP call ${tool} timed out`))
                    }
                }, 30000)
            })
        } catch (err) {
            appendDiagnostic(`MCP call ${tool} failed: ${String(err)}`)
            return null
        }
    }

    close(): void {
        if (this.process) {
            this.process.kill()
            this.process = null
        }
    }
}
