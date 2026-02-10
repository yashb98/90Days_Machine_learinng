"use strict";
// velox-api/src/routes/playground.ts
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const client_1 = require("@prisma/client");
const llmService_1 = require("../services/llmService");
const ragService_1 = require("../services/ragService");
const logger_1 = require("../utils/logger");
const router = (0, express_1.Router)();
const prisma = new client_1.PrismaClient();
router.post('/:agentId/message', async (req, res) => {
    try {
        const { agentId } = req.params;
        const { message } = req.body;
        // Load agent
        const agent = await prisma.agent.findUnique({
            where: { id: agentId },
        });
        if (!agent) {
            return res.status(404).json({ error: 'Agent not found' });
        }
        logger_1.logger.info(`Playground message for agent ${agent.name}: ${message}`);
        // Get RAG context if agent has KB
        let context = '';
        if (agent.kb_id) {
            context = await ragService_1.ragService.retrieveContext(message, agent.kb_id, 3);
            logger_1.logger.info(`Retrieved RAG context: ${context.length} chars`);
        }
        // Initialize LLM service
        const llmService = new llmService_1.LLMService();
        // Track tool calls
        const toolCalls = [];
        // Generate response with tool execution tracking
        let response = '';
        // Simple callback to capture sentences (for streaming in future)
        const onSentence = (sentence) => {
            response += sentence + ' ';
        };
        // Generate response
        await llmService.generateResponse(message, onSentence, context);
        res.json({
            response: response.trim(),
            tool_calls: toolCalls,
            context_used: !!context,
        });
    }
    catch (error) {
        logger_1.logger.error({ error }, 'Playground error');
        res.status(500).json({ error: 'Failed to process message' });
    }
});
exports.default = router;
