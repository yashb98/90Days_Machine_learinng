/*
  Warnings:

  - You are about to drop the `document_chunks` table. If the table is not empty, all the data it contains will be lost.

*/
-- AlterTable
ALTER TABLE "agents" ADD COLUMN     "kb_id" TEXT;

-- DropTable
DROP TABLE "document_chunks";

-- CreateTable
CREATE TABLE "knowledge_bases" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "embedding_model" TEXT NOT NULL DEFAULT 'text-embedding-004',
    "chunk_size" INTEGER NOT NULL DEFAULT 512,
    "chunk_overlap" INTEGER NOT NULL DEFAULT 50,
    "org_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "knowledge_bases_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "knowledge_chunks" (
    "id" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "embedding" vector(1536),
    "content_tsv" tsvector,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "kb_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "knowledge_chunks_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "knowledge_bases_org_id_idx" ON "knowledge_bases"("org_id");

-- CreateIndex
CREATE INDEX "knowledge_chunks_kb_id_idx" ON "knowledge_chunks"("kb_id");

-- CreateIndex
CREATE INDEX "knowledge_chunks_content_tsv_idx" ON "knowledge_chunks" USING GIN ("content_tsv");

-- CreateIndex
CREATE INDEX "agents_kb_id_idx" ON "agents"("kb_id");

-- AddForeignKey
ALTER TABLE "agents" ADD CONSTRAINT "agents_kb_id_fkey" FOREIGN KEY ("kb_id") REFERENCES "knowledge_bases"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "knowledge_bases" ADD CONSTRAINT "knowledge_bases_org_id_fkey" FOREIGN KEY ("org_id") REFERENCES "organizations"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "knowledge_chunks" ADD CONSTRAINT "knowledge_chunks_kb_id_fkey" FOREIGN KEY ("kb_id") REFERENCES "knowledge_bases"("id") ON DELETE CASCADE ON UPDATE CASCADE;
