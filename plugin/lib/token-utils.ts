let tokenizer: { encode: (text: string) => number[] } | null = null

function getTokenizer(): { encode: (text: string) => number[] } {
    if (!tokenizer) {
        try {
            // eslint-disable-next-line @typescript-eslint/no-require-imports
            tokenizer = require("@anthropic-ai/tokenizer")
        } catch {
            tokenizer = null
        }
    }
    return tokenizer!
}

export function estimateTokens(text: string): number {
    const tok = getTokenizer()
    if (tok) {
        try {
            return tok.encode(text).length
        } catch {
            // Fall through to estimation
        }
    }
    return Math.ceil(text.length / 4)
}

export function countTokens(text: string): number {
    return estimateTokens(text)
}
