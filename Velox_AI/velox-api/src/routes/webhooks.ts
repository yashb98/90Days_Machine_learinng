// velox-api/src/routes/voice.ts

import { Router } from 'express';
import { PrismaClient } from '@prisma/client';
import { billingService } from '../services/billingService';
import { logger } from '../utils/logger';

const router = Router();
const prisma = new PrismaClient();

/**
 * Twilio incoming call webhook
 * This endpoint receives the initial call and returns TwiML
 */
router.post('/incoming', async (req, res) => {
  try {
    const { CallSid, From, To } = req.body;

    logger.info(`Incoming call: ${CallSid} from ${From} to ${To}`);

    // Find the agent associated with this phone number
    // You'll need to store phone_number in the Agent table
    const agent = await prisma.agent.findFirst({
      where: {
        // phone_number: To, // Assuming you have this field
      },
      include: {
        org: true,
      },
    });

    if (!agent) {
      logger.warn(`No agent found for number ${To}`);
      
      const twiml = `
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
          <Say voice="Polly.Joanna">
            Sorry, this number is not configured.
          </Say>
          <Hangup/>
        </Response>
      `;
      
      return res.type('text/xml').send(twiml);
    }

    // ✅ BILLING ENFORCEMENT: Check if organization has minutes
    const hasMinutes = await billingService.hasMinutes(agent.org_id, 1);
    
    if (!hasMinutes) {
      logger.warn(`Org ${agent.org_id} has insufficient balance. Rejecting call ${CallSid}`);
      
      // Return TwiML to reject call with a polite message
      const twiml = `
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
          <Say voice="Polly.Joanna">
            We're sorry, but this service is currently unavailable due to insufficient account balance. 
            Please contact support to add more minutes and continue using this service.
          </Say>
          <Hangup/>
        </Response>
      `;
      
      return res.type('text/xml').send(twiml);
    }

    // Create conversation record
    const conversation = await prisma.conversation.create({
      data: {
        twilio_sid: CallSid,
        status: 'ACTIVE',
        agent_id: agent.id,
        start_time: new Date(),
      },
    });

    logger.info(`Created conversation ${conversation.id} for call ${CallSid}`);

    // Return TwiML with WebSocket connection
    const wsUrl = `wss://${req.get('host')}/media-stream?callSid=${CallSid}&agentId=${agent.id}`;
    
    const twiml = `
      <?xml version="1.0" encoding="UTF-8"?>
      <Response>
        <Connect>
          <Stream url="${wsUrl}">
            <Parameter name="agentId" value="${agent.id}" />
            <Parameter name="conversationId" value="${conversation.id}" />
          </Stream>
        </Connect>
      </Response>
    `;

    res.type('text/xml').send(twiml);
  } catch (error) {
    logger.error({ error }, 'Error handling incoming call');
    
    const errorTwiml = `
      <?xml version="1.0" encoding="UTF-8"?>
      <Response>
        <Say voice="Polly.Joanna">
          We're sorry, but we're experiencing technical difficulties. Please try again later.
        </Say>
        <Hangup/>
      </Response>
    `;
    
    res.type('text/xml').send(errorTwiml);
  }
});

/**
 * Call status callback
 * Twilio sends updates about call status here
 */
router.post('/status', async (req, res) => {
  try {
    const { CallSid, CallStatus } = req.body;

    logger.info(`Call status update: ${CallSid} - ${CallStatus}`);

    // Find conversation
    const conversation = await prisma.conversation.findUnique({
      where: { twilio_sid: CallSid },
    });

    if (!conversation) {
      logger.warn(`Conversation not found for CallSid: ${CallSid}`);
      return res.sendStatus(200);
    }

    // Update conversation status based on Twilio status
    let conversationStatus: 'ACTIVE' | 'COMPLETED' | 'FAILED' = 'ACTIVE';
    
    switch (CallStatus) {
      case 'completed':
        conversationStatus = 'COMPLETED';
        break;
      case 'failed':
      case 'busy':
      case 'no-answer':
      case 'canceled':
        conversationStatus = 'FAILED';
        break;
    }

    await prisma.conversation.update({
      where: { id: conversation.id },
      data: {
        status: conversationStatus,
        end_time: ['completed', 'failed', 'busy', 'no-answer', 'canceled'].includes(CallStatus)
          ? new Date()
          : undefined,
      },
    });

    logger.info(`Updated conversation ${conversation.id} status to ${conversationStatus}`);

    res.sendStatus(200);
  } catch (error) {
    logger.error({ error }, 'Error handling call status');
    res.sendStatus(500);
  }
});

/**
 * Test endpoint - simulate a call without Twilio (for development)
 */
router.post('/test', async (req, res) => {
  try {
    const { agentId, message } = req.body;

    const agent = await prisma.agent.findUnique({
      where: { id: agentId },
      include: { org: true },
    });

    if (!agent) {
      return res.status(404).json({ error: 'Agent not found' });
    }

    // Check balance
    const hasMinutes = await billingService.hasMinutes(agent.org_id, 1);
    
    if (!hasMinutes) {
      return res.status(402).json({ 
        error: 'Insufficient balance',
        message: 'Please add more minutes to continue',
      });
    }

    // Process the test message
    // ... your LLM processing logic here

    res.json({
      success: true,
      response: 'Test response',
      balance: agent.org.credit_balance,
    });
  } catch (error) {
    logger.error({ error }, 'Test endpoint error');
    res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;