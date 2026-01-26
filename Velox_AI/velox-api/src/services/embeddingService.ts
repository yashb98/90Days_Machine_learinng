// src/services/embeddingService.ts

import { logger } from "../utils/logger";

type GoogleGenAIType = {
  GoogleGenAI: new (config: { apiKey: string }) => {
    models: {
      embedContent: (params: {
        model: string;
        contents: Array<{ parts: Array<{ text: string }> }>;
      }) => Promise<{
        embeddings?: Array<{ values?: number[] }>;
      }>;
    };
  };
};

export class EmbeddingService {
  private client: InstanceType<GoogleGenAIType["GoogleGenAI"]> | null = null;

  private async getClient(): Promise<InstanceType<GoogleGenAIType["GoogleGenAI"]>> {
    if (this.client) return this.client;

    const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || "";
    if (!apiKey) {
      logger.error("❌ API key is missing. Set GEMINI_API_KEY in .env");
      throw new Error("Missing API Key");
    }

    const genai = await import("@google/genai") as GoogleGenAIType;
    this.client = new genai.GoogleGenAI({ apiKey });
    return this.client;
  }

  async getEmbedding(text: string): Promise<number[] | null> {
    try {
      if (!text || text.trim().length === 0) {
        logger.warn("⚠️ Empty text provided for embedding");
        return null;
      }

      // Truncate very long texts to avoid memory issues
      const maxLength = 10000;
      const truncatedText = text.length > maxLength ? text.substring(0, maxLength) : text;

      if (text.length > maxLength) {
        logger.warn(`⚠️ Text truncated from ${text.length} to ${maxLength} characters`);
      }

      const client = await this.getClient();

      logger.info(`📊 Generating embedding for text (${truncatedText.length} chars)...`);

      const result = await client.models.embedContent({
        model: "text-embedding-004",
        contents: [{ parts: [{ text: truncatedText }] }],
      });
      
      const values = result.embeddings?.[0]?.values;

      if (!values || !Array.isArray(values)) {
        logger.error("❌ Invalid embedding response structure");
        return null;
      }

      logger.info(`✅ Generated embedding: ${values.length} dimensions`);
      return values;

    } catch (error: any) {
      logger.error({ error: error.message }, "Error generating embedding");
      return null;
    }
  }

  /**
   * Generate embeddings with delay to avoid rate limits and memory issues
   */
  async getEmbeddings(texts: string[]): Promise<(number[] | null)[]> {
    try {
      const embeddings: (number[] | null)[] = [];
      
      logger.info(`📊 Generating ${texts.length} embeddings sequentially...`);

      for (let i = 0; i < texts.length; i++) {
        logger.info(`Processing ${i + 1}/${texts.length}...`);
        
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
      logger.info(`✅ Generated ${successCount}/${texts.length} embeddings`);
      
      return embeddings;

    } catch (error: any) {
      logger.error({ error: error.message }, "Error generating batch embeddings");
      return texts.map(() => null);
    }
  }
}

export const embeddingService = new EmbeddingService();