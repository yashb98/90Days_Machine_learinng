"use strict";
// src/services/embeddingService.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.embeddingService = exports.EmbeddingService = void 0;
const logger_1 = require("../utils/logger");
class EmbeddingService {
    constructor() {
        this.client = null;
    }
    async getClient() {
        if (this.client)
            return this.client;
        const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || "";
        if (!apiKey) {
            logger_1.logger.error("❌ API key is missing. Set GEMINI_API_KEY in .env");
            throw new Error("Missing API Key");
        }
        const genai = await import("@google/genai");
        this.client = new genai.GoogleGenAI({ apiKey });
        return this.client;
    }
    async getEmbedding(text) {
        try {
            if (!text || text.trim().length === 0) {
                logger_1.logger.warn("⚠️ Empty text provided for embedding");
                return null;
            }
            // Truncate very long texts to avoid memory issues
            const maxLength = 10000;
            const truncatedText = text.length > maxLength ? text.substring(0, maxLength) : text;
            if (text.length > maxLength) {
                logger_1.logger.warn(`⚠️ Text truncated from ${text.length} to ${maxLength} characters`);
            }
            const client = await this.getClient();
            logger_1.logger.info(`📊 Generating embedding for text (${truncatedText.length} chars)...`);
            const result = await client.models.embedContent({
                model: "text-embedding-004",
                contents: [{ parts: [{ text: truncatedText }] }],
            });
            const values = result.embeddings?.[0]?.values;
            if (!values || !Array.isArray(values)) {
                logger_1.logger.error("❌ Invalid embedding response structure");
                return null;
            }
            logger_1.logger.info(`✅ Generated embedding: ${values.length} dimensions`);
            return values;
        }
        catch (error) {
            logger_1.logger.error({ error: error.message }, "Error generating embedding");
            return null;
        }
    }
    /**
     * Generate embeddings with delay to avoid rate limits and memory issues
     */
    async getEmbeddings(texts) {
        try {
            const embeddings = [];
            logger_1.logger.info(`📊 Generating ${texts.length} embeddings sequentially...`);
            for (let i = 0; i < texts.length; i++) {
                logger_1.logger.info(`Processing ${i + 1}/${texts.length}...`);
                const embedding = await this.getEmbedding(texts[i]);
                embeddings.push(embedding);
                // Add small delay to prevent memory buildup
                if (i < texts.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
                // Force garbage collection hint
                if (global.gc && i % 5 === 0) {
                    global.gc();
                }
            }
            const successCount = embeddings.filter(e => e !== null).length;
            logger_1.logger.info(`✅ Generated ${successCount}/${texts.length} embeddings`);
            return embeddings;
        }
        catch (error) {
            logger_1.logger.error({ error: error.message }, "Error generating batch embeddings");
            return texts.map(() => null);
        }
    }
}
exports.EmbeddingService = EmbeddingService;
exports.embeddingService = new EmbeddingService();
