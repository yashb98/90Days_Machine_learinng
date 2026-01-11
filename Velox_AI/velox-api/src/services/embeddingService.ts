// 1. Use 'import type' to prevent runtime crash on startup
import { GoogleGenAI } from "@google/genai";

import { logger } from "../utils/logger";

export class EmbeddingService {
  // Store the client instance
  private client: GoogleGenAI | null = null;

  // 2. Helper to load the SDK dynamically (Lazy Loading)
  private async getClient(): Promise<GoogleGenAI> {
    if (this.client) return this.client;

    const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || "";
    if (!apiKey) {
      logger.error("❌ API key is missing. Set GEMINI_API_KEY in .env");
      throw new Error("Missing API Key");
    }

    // ⚠️ DYNAMIC IMPORT: Fixes the "require" error
    const { GoogleGenAI } = await import("@google/genai");
    
    this.client = new GoogleGenAI({ apiKey });
    return this.client;
  }

  async getEmbedding(text: string): Promise<number[] | null> {
    try {
      if (!text || text.trim().length === 0) {
        logger.warn("⚠️ Empty text provided for embedding");
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
        logger.error("❌ Invalid embedding response structure");
        return null;
      }
      
      return values;

    } catch (error: any) {
      logger.error({ error: error.message }, "Error generating embedding");
      return null;
    }
  }
}