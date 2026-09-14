export interface SessionFile {
    _id: string
    title: string
    startedAt: string
    updatedAt: string
    compactedAt: string | null
    parentId: string | null
    subSessions: string[]
    goal: string | null
    turns: Turn[]
    permissions: PermissionEvent[]
    errors: SessionError[]
    editImpact: EditImpact[]
    importanceScores: ImportanceSnapshot | null
    interconnectedness: InterconnectednessSnapshot | null
}

export interface Turn {
    role: "user" | "assistant"
    messageId: string
    timestamp: string
    parts: Part[]
    error?: TurnError
    importanceScores?: {
        filesEdited: Record<string, number>
        functionsEdited: Record<string, number>
    }
}

export interface TurnError {
    name: string
    message: string
    statusCode?: number
    isRetryable?: boolean
    type?: string
}

export interface SessionError {
    timestamp: string
    name: string
    message: string
    statusCode?: number
    sessionID?: string
}

export interface Part {
    type: "text" | "thinking" | "tool" | "file" | "subtask"
    text?: string
    toolName?: string
    toolInput?: Record<string, unknown>
    toolOutput?: string
    isError?: boolean
    fileName?: string
    mime?: string
    agent?: string
    prompt?: string
    description?: string
    editMetadata?: EditMetadata
}

export interface EditMetadata {
    reason: string
    contributesToGoal: string
    success: boolean
    tradeoffs?: string
}

export interface EditImpact {
    file: string
    function: string | null
    timestamp: string
    change: string
    reason: string
    impact: string
    tested: boolean
    testResult?: string
}

export interface ImportanceSnapshot {
    files: Record<string, number>
    functions: Record<string, number>
    mlPredictions: {
        files: Record<string, number>
        functions: Record<string, number>
    }
}

export interface InterconnectednessSnapshot {
    files: Record<string, Record<string, { weight: number; relation: string; context?: string }>>
    functions: Record<string, Record<string, { weight: number; relation: string }>>
}

export interface PermissionEvent {
    type: "asked" | "replied"
    timestamp: string
    permission: string
    command?: string
    question?: string
    allowed?: boolean
}

export interface IndexEntry {
    id: string
    title: string
    startedAt: string
    updatedAt: string
    parentId: string | null
}

export interface SessionIndex {
    sessions: IndexEntry[]
}

export interface SessionInfo {
    id: string
    title: string
    createdAt: string
    updatedAt: string
    parentId: string | null
}
