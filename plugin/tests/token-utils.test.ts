import { describe, it, expect } from "vitest"
import { countTokens, estimateTokens } from "../lib/token-utils"

describe("countTokens", () => {
    it("returns 0 for empty string", () => {
        expect(countTokens("")).toBe(0)
    })

    it("returns positive number for non-empty string", () => {
        expect(countTokens("hello world")).toBeGreaterThan(0)
    })

    it("longer text has more tokens", () => {
        const short = "hello"
        const long = "hello world this is a much longer string with more words"
        expect(countTokens(long)).toBeGreaterThan(countTokens(short))
    })
})

describe("estimateTokens", () => {
    it("returns 0 for empty string", () => {
        expect(estimateTokens("")).toBe(0)
    })

    it("returns positive number for text", () => {
        expect(estimateTokens("test")).toBeGreaterThan(0)
    })
})
