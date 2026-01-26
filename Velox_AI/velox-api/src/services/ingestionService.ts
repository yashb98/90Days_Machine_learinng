// src/services/ingestionService.ts

import { PrismaClient } from "@prisma/client";
import { embeddingService } from "./embeddingService";
import { logger } from "../utils/logger";

const prisma = new PrismaClient();

interface ChunkMetadata {
  source?: string;
  page?: number;
  section?: string;
  [key: string]: any;
}

export class IngestionService {
  /**
   * Split text into chunks with overlap
   */
  private splitIntoChunks(
    text: string,
    chunkSize: number = 512,
    overlap: number = 50
  ): string[] {
    const chunks: string[] = [];
    let start = 0;

    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      const chunk = text.slice(start, end).trim();
      
      if (chunk.length > 0) {
        chunks.push(chunk);
      }

      start = end - overlap;
    }

    return chunks;
  }

  /**
   * Ingest a single document into the knowledge base
   */
  async ingestDocument(
    kbId: string,
    content: string,
    metadata: ChunkMetadata = {}
  ): Promise<{ success: boolean; chunksCreated: number }> {
    try {
      logger.info(`📄 Starting ingestion for KB: ${kbId}`);

      // Get KB settings
      const kb = await prisma.knowledgeBase.findUnique({
        where: { id: kbId },
      });

      if (!kb) {
        throw new Error("Knowledge base not found");
      }

      // Split into chunks
      const chunks = this.splitIntoChunks(
        content,
        kb.chunk_size,
        kb.chunk_overlap
      );

      logger.info(`✂️ Split into ${chunks.length} chunks`);

      // Generate embeddings for all chunks
      const embeddings = await embeddingService.getEmbeddings(chunks);

      // Insert chunks into database
      let successCount = 0;
      for (let i = 0; i < chunks.length; i++) {
        const embedding = embeddings[i];
        
        if (!embedding) {
          logger.warn(`⚠️ Skipping chunk ${i + 1} (embedding failed)`);
          continue;
        }

        // Convert embedding array to pgvector format
        const embeddingStr = `[${embedding.join(",")}]`;

        await prisma.$executeRaw`
          INSERT INTO "knowledge_chunks" (id, content, embedding, metadata, kb_id, created_at, updated_at)
          VALUES (
            gen_random_uuid(),
            ${chunks[i]},
            ${embeddingStr}::vector,
            ${JSON.stringify(metadata)}::jsonb,
            ${kbId},
            NOW(),
            NOW()
          )
        `;

        successCount++;
      }

      logger.info(`✅ Ingestion complete: ${successCount}/${chunks.length} chunks created`);

      return {
        success: true,
        chunksCreated: successCount,
      };

    } catch (error: any) {
      logger.error({ error: error.message }, "Ingestion failed");
      return {
        success: false,
        chunksCreated: 0,
      };
    }
  }

  /**
   * Ingest multiple documents in batch
   */
  async ingestDocuments(
    kbId: string,
    documents: Array<{ content: string; metadata?: ChunkMetadata }>
  ): Promise<{ success: boolean; totalChunks: number }> {
    try {
      let totalChunks = 0;

      for (const doc of documents) {
        const result = await this.ingestDocument(kbId, doc.content, doc.metadata);
        totalChunks += result.chunksCreated;
      }

      return {
        success: true,
        totalChunks,
      };

    } catch (error: any) {
      logger.error({ error: error.message }, "Batch ingestion failed");
      return {
        success: false,
        totalChunks: 0,
      };
    }
  }
}

export const ingestionService = new IngestionService();