import { GoogleGenerativeAI } from "@google/generative-ai";
import { logger } from "../app";

// Use the same environment variable name as llmService for consistency
const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || "";
if (!apiKey) {
  logger.warn("⚠️ GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables");
}

const genAI = new GoogleGenerativeAI(apiKey);

export class EmbeddingService {
  private model = genAI.getGenerativeModel({ model: "text-embedding-004" });

  async getEmbedding(text: string): Promise<number[] | null> {
    try {
      if (!text || text.trim().length === 0) {
        logger.warn("Empty text provided for embedding");
        return null;
      }

      if (!apiKey) {
        logger.error("API key is missing. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.");
        return null;
      }

      const result = await this.model.embedContent(text);
      
      const embedding = result.embedding;
      
      if (!embedding || !embedding.values || !Array.isArray(embedding.values)) {
        logger.error({ embedding }, "Invalid embedding response structure");
        return null;
      }
      
      return embedding.values;
    } catch (error: any) {
      const errorDetails = {
        message: error?.message || String(error),
        stack: error?.stack,
        name: error?.name,
        ...(error?.code && { code: error.code })
      };
      logger.error({ error: errorDetails }, "Error generating embedding");
      return null;
    }
  }
}