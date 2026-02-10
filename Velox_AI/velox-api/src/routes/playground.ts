// velox-api/src/routes/playground.ts

import { Router } from 'express';
import { PrismaClient } from '@prisma/client';
import { LLMService } from '../services/llmService';
import { ragService } from '../services/ragService';
import { logger } from '../utils/logger';

const router = Router();
const prisma = new PrismaClient();

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

    logger.info(`Playground message for agent ${agent.name}: ${message}`);

    // Get RAG context if agent has KB
    let context = '';
    if (agent.kb_id) {
      context = await ragService.retrieveContext(message, agent.kb_id, 3);
      logger.info(`Retrieved RAG context: ${context.length} chars`);
    }

    // Initialize LLM service
    const llmService = new LLMService();
    
    // Track tool calls
    const toolCalls: any[] = [];
    
    // Generate response with tool execution tracking
    let response = '';
    
    // Simple callback to capture sentences (for streaming in future)
    const onSentence = (sentence: string) => {
      response += sentence + ' ';
    };

    // Generate response
    await llmService.generateResponse(message, onSentence, context);

    res.json({
      response: response.trim(),
      tool_calls: toolCalls,
      context_used: !!context,
    });

  } catch (error: any) {
    logger.error({ error }, 'Playground error');
    res.status(500).json({ error: 'Failed to process message' });
  }
});

export default router;