import { logger } from "../utils/logger";

// Type for the dynamically imported GoogleGenAI
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
  // Store the client instance
  private client: InstanceType<GoogleGenAIType["GoogleGenAI"]> | null = null;

  // Helper to load the SDK dynamically (Lazy Loading)
  private async getClient(): Promise<InstanceType<GoogleGenAIType["GoogleGenAI"]>> {
    if (this.client) return this.client;

    const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || "";
    if (!apiKey) {
      logger.error("❌ API key is missing. Set GEMINI_API_KEY in .env");
      throw new Error("Missing API Key");
    }

    // DYNAMIC IMPORT: Fixes the "require" error
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

      // Get the dynamically loaded client
      const client = await this.getClient();

      // Use NEW SDK Syntax for embeddings
      const result = await client.models.embedContent({
        model: "text-embedding-004",
        contents: [{ parts: [{ text }] }],
      });
      
      // The new SDK returns 'embeddings[0].values'
      const values = result.embeddings?.[0]?.values;

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

