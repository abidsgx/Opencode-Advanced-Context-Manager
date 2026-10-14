import { CtxManagerPlugin } from "./lib/plugin"
export { resolveMcpConfig, getServerPaths } from "./lib/server-resolver"
export { loadConfig } from "./lib/config"
export { McpClient } from "./lib/mcp-client"

export default {
  id: "ctx-manager",
  server: CtxManagerPlugin,
}
