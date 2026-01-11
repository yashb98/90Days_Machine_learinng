"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.EmbeddingService = void 0;
const logger_1 = require("../utils/logger");
class EmbeddingService {
    constructor() {
        // Store the client instance
        this.client = null;
    }
    // 2. Helper to load the SDK dynamically (Lazy Loading)
    async getClient() {
        if (this.client)
            return this.client;
        const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || "";
        if (!apiKey) {
            logger_1.logger.error("❌ API key is missing. Set GEMINI_API_KEY in .env");
            throw new Error("Missing API Key");
        }
        // ⚠️ DYNAMIC IMPORT: Fixes the "require" error
        const { GoogleGenAI } = await import("@google/genai");
        this.client = new GoogleGenAI({ apiKey });
        return this.client;
    }
    async getEmbedding(text) {
        try {
            if (!text || text.trim().length === 0) {
                logger_1.logger.warn("⚠️ Empty text provided for embedding");
                return null;
            }
            // 3. Get the dynamically loaded client
            const client = await this.getClient();
            // 4. Use NEW SDK Syntax for embeddings
            const result = await client.models.embedContent({
                model: "text-embedding-004",
                contents: [{ parts: [{ text }] }],
            });
            // The new SDK returns 'embedding.values'
            const values = result.embedding?.values;
            if (!values || !Array.isArray(values)) {
                logger_1.logger.error("❌ Invalid embedding response structure");
                return null;
            }
            return values;
        }
        catch (error) {
            logger_1.logger.error({ error: error.message }, "Error generating embedding");
            return null;
        }
    }
}
exports.EmbeddingService = EmbeddingService;
