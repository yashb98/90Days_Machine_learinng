"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const multer_1 = __importDefault(require("multer"));
const textsplitters_1 = require("@langchain/textsplitters");
const embeddingService_1 = require("../services/embeddingService");
const db_1 = require("../db"); // Assuming you have a DB connection exported
const logger_1 = require("../utils/logger");
// pdf-parse is a CommonJS module, need to handle it properly
// eslint-disable-next-line @typescript-eslint/no-var-requires
const pdfParseModule = require("pdf-parse");
const PDFParse = pdfParseModule.PDFParse || pdfParseModule;
const router = (0, express_1.Router)();
const upload = (0, multer_1.default)({
    storage: multer_1.default.memoryStorage(),
    limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limit
    }
}); // Store files in RAM temporarily
const embeddingService = new embeddingService_1.EmbeddingService();
// Wrapper to handle multer errors
const uploadMiddleware = (req, res, next) => {
    upload.single("file")(req, res, (err) => {
        if (err instanceof multer_1.default.MulterError) {
            if (err.code === "LIMIT_FILE_SIZE") {
                return res.status(400).json({ error: "File too large. Maximum size is 10MB." });
            }
            if (err.code === "LIMIT_UNEXPECTED_FILE") {
                return res.status(400).json({
                    error: `Unexpected field: ${err.field}. Expected field name: "file"`
                });
            }
            return res.status(400).json({ error: `Multer error: ${err.message}` });
        }
        if (err) {
            return res.status(400).json({ error: err.message || "File upload error" });
        }
        next();
    });
};
router.post("/upload", uploadMiddleware, async (req, res) => {
    try {
        if (!req.file) {
            res.status(400).json({ error: "No file uploaded. Please use field name 'file' in your multipart form data." });
            return;
        }
        logger_1.logger.info(` Processing file: ${req.file.originalname}`);
        // 1. Parse Text from PDF
        const pdfParser = new PDFParse({ data: req.file.buffer });
        const data = await pdfParser.getText();
        const fullText = data.text;
        if (!fullText || fullText.trim().length === 0) {
            logger_1.logger.warn("PDF appears to be empty or unreadable");
            res.status(400).json({ error: "PDF file appears to be empty or unreadable" });
            return;
        }
        // 2. Split Text into Chunks (Semantic Splitting)
        const splitter = new textsplitters_1.RecursiveCharacterTextSplitter({
            chunkSize: 500, // ~100 words per chunk
            chunkOverlap: 50, // Context overlap
        });
        const chunks = await splitter.createDocuments([fullText]);
        logger_1.logger.info(`🔪 Split into ${chunks.length} chunks. Generating embeddings...`);
        // 3. Generate Embeddings & Insert into DB
        let savedCount = 0;
        let errorCount = 0;
        // Process in parallel (batches of 5 to avoid rate limits)
        for (const chunk of chunks) {
            try {
                const vector = await embeddingService.getEmbedding(chunk.pageContent);
                if (vector) {
                    // PGVector format: '[0.1, 0.2, ...]'
                    const vectorString = `[${vector.join(",")}]`;
                    await db_1.pool.query(`INSERT INTO document_chunks (content, embedding, metadata) VALUES ($1, $2, $3)`, [chunk.pageContent, vectorString, { source: req.file.originalname }]);
                    savedCount++;
                }
                else {
                    errorCount++;
                    logger_1.logger.warn("Failed to generate embedding for a chunk");
                }
            }
            catch (chunkError) {
                errorCount++;
                logger_1.logger.error({
                    error: chunkError?.message || String(chunkError),
                    stack: chunkError?.stack
                }, "Error processing chunk");
            }
        }
        if (savedCount === 0) {
            logger_1.logger.error("No chunks were successfully saved");
            res.status(500).json({
                error: "Failed to process document. No chunks were saved.",
                details: errorCount > 0 ? `${errorCount} chunks failed to process` : undefined
            });
            return;
        }
        logger_1.logger.info(` Successfully indexed ${savedCount} chunks.`);
        res.json({
            status: "success",
            chunks: savedCount,
            ...(errorCount > 0 && { warnings: `${errorCount} chunks failed to process` })
        });
    }
    catch (error) {
        // Properly serialize error for logging
        const errorDetails = {
            message: error?.message || String(error),
            stack: error?.stack,
            name: error?.name,
            ...(error?.code && { code: error.code })
        };
        logger_1.logger.error({ error: errorDetails }, "Ingestion failed");
        res.status(500).json({
            error: "Internal Server Error",
            message: error?.message || "An unexpected error occurred during document processing"
        });
    }
});
exports.default = router;
