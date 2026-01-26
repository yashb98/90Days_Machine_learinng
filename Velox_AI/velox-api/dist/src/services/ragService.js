"use strict";
// src/services/ragService.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.ragService = exports.RAGService = void 0;
const hybridSearchService_1 = require("./hybridSearchService");
const embeddingService_1 = require("./embeddingService"); // ✅ Now exports the instance
const logger_1 = require("../utils/logger");
class RAGService {
    async retrieveContext(query, kbId, limit = 3) {
        try {
            // ✅ Use getEmbedding instead of generateEmbedding
            const embedding = await embeddingService_1.embeddingService.getEmbedding(query);
            if (!embedding) {
                logger_1.logger.error("Failed to generate embedding for query");
                return "";
            }
            const results = await hybridSearchService_1.hybridSearchService.smartSearch(query, embedding, kbId, limit);
            if (results.length === 0) {
                logger_1.logger.warn("No context found");
                return "";
            }
            const context = results
                .map((r, i) => {
                const icon = r.source === "both" ? "★" : r.source === "keyword" ? "K" : "S";
                return `[${icon}${i + 1}] ${r.content}`;
            })
                .join("\n\n");
            logger_1.logger.info({ count: results.length }, "✅ Context retrieved");
            return context;
        }
        catch (error) {
            logger_1.logger.error({ error: error.message }, "RAG retrieval failed");
            return "";
        }
    }
}
exports.RAGService = RAGService;
// ✅ Export singleton instance
exports.ragService = new RAGService();
