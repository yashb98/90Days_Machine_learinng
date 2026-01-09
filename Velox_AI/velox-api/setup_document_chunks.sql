-- Setup script for document_chunks table with pgvector support
-- Run this script to create the table needed for document embeddings

-- Enable pgvector extension (required for vector embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- CreateTable: document_chunks
CREATE TABLE IF NOT EXISTS "document_chunks" (
    "id" SERIAL PRIMARY KEY,
    "content" TEXT NOT NULL,
    "embedding" vector(768), -- Gemini text-embedding-004 uses 768 dimensions
    "metadata" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- CreateIndex: For vector similarity search
CREATE INDEX IF NOT EXISTS "document_chunks_embedding_idx" 
ON "document_chunks" 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- CreateIndex: For metadata queries
CREATE INDEX IF NOT EXISTS "document_chunks_metadata_idx" 
ON "document_chunks" 
USING GIN (metadata);

-- CreateIndex: For content search
CREATE INDEX IF NOT EXISTS "document_chunks_content_idx" 
ON "document_chunks" 
USING GIN (to_tsvector('english', content));

-- Verify table was created
SELECT 
    table_name, 
    column_name, 
    data_type 
FROM information_schema.columns 
WHERE table_name = 'document_chunks'
ORDER BY ordinal_position;
