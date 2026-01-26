// src/services/ragService.ts

import { hybridSearchService } from "./hybridSearchService";
import { embeddingService } from "./embeddingService"; // ✅ Now exports the instance
import { logger } from "../utils/logger";

export class RAGService {
  async retrieveContext(
    query: string,
    kbId: string,
    limit: number = 3
  ): Promise<string> {
    try {
      // ✅ Use getEmbedding instead of generateEmbedding
      const embedding = await embeddingService.getEmbedding(query);

      if (!embedding) {
        logger.error("Failed to generate embedding for query");
        return "";
      }

      const results = await hybridSearchService.smartSearch(
        query,
        embedding,
        kbId,
        limit
      );

      if (results.length === 0) {
        logger.warn("No context found");
        return "";
      }

      const context = results
        .map((r, i) => {
          const icon = r.source === "both" ? "★" : r.source === "keyword" ? "K" : "S";
          return `[${icon}${i + 1}] ${r.content}`;
        })
        .join("\n\n");

      logger.info({ count: results.length }, "✅ Context retrieved");
      return context;

    } catch (error: any) {
      logger.error({ error: error.message }, "RAG retrieval failed");
      return "";
    }
  }
}

// ✅ Export singleton instance
export const ragService = new RAGService();