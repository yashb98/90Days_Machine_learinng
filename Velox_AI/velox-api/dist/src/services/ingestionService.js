"use strict";
// src/services/ingestionService.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.ingestionService = exports.IngestionService = void 0;
const client_1 = require("@prisma/client");
const embeddingService_1 = require("./embeddingService");
const logger_1 = require("../utils/logger");
const prisma = new client_1.PrismaClient();
class IngestionService {
    /**
     * Split text into chunks with overlap
     */
    splitIntoChunks(text, chunkSize = 512, overlap = 50) {
        const chunks = [];
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
    async ingestDocument(kbId, content, metadata = {}) {
        try {
            logger_1.logger.info(`📄 Starting ingestion for KB: ${kbId}`);
            // Get KB settings
            const kb = await prisma.knowledgeBase.findUnique({
                where: { id: kbId },
            });
            if (!kb) {
                throw new Error("Knowledge base not found");
            }
            // Split into chunks
            const chunks = this.splitIntoChunks(content, kb.chunk_size, kb.chunk_overlap);
            logger_1.logger.info(`✂️ Split into ${chunks.length} chunks`);
            // Generate embeddings for all chunks
            const embeddings = await embeddingService_1.embeddingService.getEmbeddings(chunks);
            // Insert chunks into database
            let successCount = 0;
            for (let i = 0; i < chunks.length; i++) {
                const embedding = embeddings[i];
                if (!embedding) {
                    logger_1.logger.warn(`⚠️ Skipping chunk ${i + 1} (embedding failed)`);
                    continue;
                }
                // Convert embedding array to pgvector format
                const embeddingStr = `[${embedding.join(",")}]`;
                await prisma.$executeRaw `
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
            logger_1.logger.info(`✅ Ingestion complete: ${successCount}/${chunks.length} chunks created`);
            return {
                success: true,
                chunksCreated: successCount,
            };
        }
        catch (error) {
            logger_1.logger.error({ error: error.message }, "Ingestion failed");
            return {
                success: false,
                chunksCreated: 0,
            };
        }
    }
    /**
     * Ingest multiple documents in batch
     */
    async ingestDocuments(kbId, documents) {
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
        }
        catch (error) {
            logger_1.logger.error({ error: error.message }, "Batch ingestion failed");
            return {
                success: false,
                totalChunks: 0,
            };
        }
    }
}
exports.IngestionService = IngestionService;
exports.ingestionService = new IngestionService();
