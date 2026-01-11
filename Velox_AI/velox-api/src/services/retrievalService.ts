import { pool } from "../db";
import { EmbeddingService } from "./embeddingService";
import { logger } from "../utils/logger";

export class RetrievalService {
  private embeddingService: EmbeddingService;

  constructor() {
    this.embeddingService = new EmbeddingService();
  }

  async search(query: string, limit: number = 3): Promise<string> {
    try {
      // 1. Convert the user's question into a Vector
      const queryEmbedding = await this.embeddingService.getEmbedding(query);
      
      if (!queryEmbedding) {
        return "";
      }

      const vectorString = `[${queryEmbedding.join(",")}]`;

      // 2. Run the Similarity Search (Cosine Distance)
      // Note: We use <=> operator for cosine distance. 
      // 1 - distance gives us similarity (1.0 is identical, 0.0 is opposite).
      const sql = `
        SELECT content, 1 - (embedding <=> $1) as similarity
        FROM document_chunks
        WHERE 1 - (embedding <=> $1) > 0.3  -- Threshold: Ignore irrelevant chunks
        ORDER BY similarity DESC
        LIMIT $2;
      `;

      const { rows } = await pool.query(sql, [vectorString, limit]);

      if (rows.length === 0) {
        logger.info("❌ No relevant context found in DB.");
        return "";
      }

      logger.info(`✅ Found ${rows.length} relevant chunks. (Top Score: ${rows[0].similarity.toFixed(2)})`);
      
      // 3. Combine the chunks into a single text block
      return rows.map((r) => r.content).join("\n---\n");

    } catch (error) {
      logger.error({ error }, "Error during vector search");
      return "";
    }
  }
}