"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.EmbeddingService = void 0;
const generative_ai_1 = require("@google/generative-ai");
const logger_1 = require("../utils/logger");
// Use the same environment variable name as llmService for consistency
const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || "";
if (!apiKey) {
    logger_1.logger.warn("⚠️ GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables");
}
const genAI = new generative_ai_1.GoogleGenerativeAI(apiKey);
class EmbeddingService {
    constructor() {
        this.model = genAI.getGenerativeModel({ model: "text-embedding-004" });
    }
    async getEmbedding(text) {
        try {
            if (!text || text.trim().length === 0) {
                logger_1.logger.warn("Empty text provided for embedding");
                return null;
            }
            if (!apiKey) {
                logger_1.logger.error("API key is missing. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.");
                return null;
            }
            const result = await this.model.embedContent(text);
            const embedding = result.embedding;
            if (!embedding || !embedding.values || !Array.isArray(embedding.values)) {
                logger_1.logger.error({ embedding }, "Invalid embedding response structure");
                return null;
            }
            return embedding.values;
        }
        catch (error) {
            const errorDetails = {
                message: error?.message || String(error),
                stack: error?.stack,
                name: error?.name,
                ...(error?.code && { code: error.code })
            };
            logger_1.logger.error({ error: errorDetails }, "Error generating embedding");
            return null;
        }
    }
}
exports.EmbeddingService = EmbeddingService;
