// src/services/hybridSearchService.ts

import { PrismaClient } from "@prisma/client";
import { logger } from "../utils/logger";

const prisma = new PrismaClient();

interface SearchResult {
  id: string;
  content: string;
  metadata: any;
  score: number;
  rank: number;
  source: "keyword" | "semantic" | "both";
}

/**
 * Reciprocal Rank Fusion (RRF)
 * Formula: RRF(d) = Σ 1/(k + rank(d))
 */
function reciprocalRankFusion(
  keywordResults: any[],
  semanticResults: any[],
  k: number = 60
): SearchResult[] {
  const rrfScores = new Map<string, { score: number; doc: any; sources: Set<string> }>();

  // Process keyword results
  keywordResults.forEach((doc, index) => {
    const rank = index + 1;
    const score = 1 / (k + rank);
    
    if (!rrfScores.has(doc.id)) {
      rrfScores.set(doc.id, { score: 0, doc, sources: new Set() });
    }
    
    const entry = rrfScores.get(doc.id)!;
    entry.score += score;
    entry.sources.add("keyword");
  });

  // Process semantic results
  semanticResults.forEach((doc, index) => {
    const rank = index + 1;
    const score = 1 / (k + rank);
    
    if (!rrfScores.has(doc.id)) {
      rrfScores.set(doc.id, { score: 0, doc, sources: new Set() });
    }
    
    const entry = rrfScores.get(doc.id)!;
    entry.score += score;
    entry.sources.add("semantic");
  });

  // Sort by RRF score
  const results = Array.from(rrfScores.entries())
    .map(([id, { score, doc, sources }]) => ({
      id,
      content: doc.content,
      metadata: doc.metadata,
      score,
      rank: 0,
      source: (sources.size === 2 ? "both" : 
              sources.has("keyword") ? "keyword" : "semantic") as "keyword" | "semantic" | "both",
    }))
    .sort((a, b) => b.score - a.score);

  results.forEach((result, index) => {
    result.rank = index + 1;
  });

  return results;
}

export class HybridSearchService {
  /**
   * Keyword search using PostgreSQL Full-Text Search
   */
  private async keywordSearch(
    query: string,
    kbId: string,
    limit: number = 10
  ): Promise<any[]> {
    const startTime = Date.now();

    try {
      const results = await prisma.$queryRaw<any[]>`
        SELECT 
          id,
          content,
          metadata,
          ts_rank(content_tsv, plainto_tsquery('english', ${query})) as relevance
        FROM "knowledge_chunks"
        WHERE 
          kb_id = ${kbId}
          AND content_tsv @@ plainto_tsquery('english', ${query})
        ORDER BY relevance DESC
        LIMIT ${limit}
      `;

      const duration = Date.now() - startTime;
      logger.info(`🔍 Keyword search: ${duration}ms, ${results.length} results`);

      return results;
    } catch (error) {
      logger.error({ error }, "Keyword search failed");
      return [];
    }
  }

  /**
   * Semantic search using pgvector
   */
  private async semanticSearch(
    embedding: number[],
    kbId: string,
    limit: number = 10
  ): Promise<any[]> {
    const startTime = Date.now();

    try {
      const embeddingStr = `[${embedding.join(",")}]`;

      const results = await prisma.$queryRaw<any[]>`
        SELECT 
          id,
          content,
          metadata,
          1 - (embedding <=> ${embeddingStr}::vector) as similarity
        FROM "knowledge_chunks"
        WHERE kb_id = ${kbId}
        ORDER BY embedding <=> ${embeddingStr}::vector
        LIMIT ${limit}
      `;

      const duration = Date.now() - startTime;
      logger.info(`🧠 Semantic search: ${duration}ms, ${results.length} results`);

      return results;
    } catch (error) {
      logger.error({ error }, "Semantic search failed");
      return [];
    }
  }

  /**
   * Hybrid search with RRF
   */
  async search(
    query: string,
    embedding: number[],
    kbId: string,
    limit: number = 5
  ): Promise<SearchResult[]> {
    const searchLimit = limit * 2;

    logger.info(`🔎 Hybrid search: "${query}" in KB: ${kbId}`);

    const [keywordResults, semanticResults] = await Promise.all([
      this.keywordSearch(query, kbId, searchLimit),
      this.semanticSearch(embedding, kbId, searchLimit),
    ]);

    const fusedResults = reciprocalRankFusion(keywordResults, semanticResults);
    const finalResults = fusedResults.slice(0, limit);

    logger.info({
      total: finalResults.length,
      keyword: finalResults.filter(r => r.source === "keyword").length,
      semantic: finalResults.filter(r => r.source === "semantic").length,
      both: finalResults.filter(r => r.source === "both").length,
    }, "✅ Hybrid search completed");

    return finalResults;
  }

  /**
   * Detect specific identifiers in query
   */
  private hasSpecificIdentifiers(query: string): boolean {
    const patterns = [
      /\b\d{3,}\b/,
      /\b[A-Z]{2,}\d+\b/,
      /order\s+\d+/i,
      /ticket\s+\d+/i,
    ];

    return patterns.some(pattern => pattern.test(query));
  }

  /**
   * Smart search adapts to query type
   */
  async smartSearch(
    query: string,
    embedding: number[],
    kbId: string,
    limit: number = 5
  ): Promise<SearchResult[]> {
    const hasIds = this.hasSpecificIdentifiers(query);
    
    if (hasIds) {
      logger.info("🎯 Detected specific IDs - boosting keyword search");
    } else {
      logger.info("💭 Conceptual query - balanced hybrid search");
    }

    return this.search(query, embedding, kbId, limit);
  }
}

export const hybridSearchService = new HybridSearchService();