export interface ContextPriority {
    messageId: string
    score: number
    reason: string
}

export interface CompressionBlock {
    blockId: number
    runId: number
    active: boolean
    summary: string
    summaryTokens: number
    compressedTokens: number
    topic: string
    startId: string
    endId: string
    createdAt: number
}

export interface PrunedMessageEntry {
    tokenCount: number
    blockIds: number[]
}

export interface MessageIdState {
    byRawId: Map<string, string>
    byRef: Map<string, string>
    nextRef: number
}

export interface SessionStats {
    pruneTokenCounter: number
    totalPruneTokens: number
    filesPruned: number
    filesPreserved: number
}

export interface CtxSessionState {
    sessionId: string | null
    currentTurn: number
    priority: Map<string, ContextPriority>
    compressionBlocks: Map<number, CompressionBlock>
    prunedMessages: Map<string, PrunedMessageEntry>
    messageIds: MessageIdState
    stats: SessionStats
    modelContextLimit: number | undefined
    systemPromptTokens: number | undefined
}
