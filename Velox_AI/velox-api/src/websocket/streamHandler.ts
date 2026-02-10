// velox-api/src/websocket/streamHandler.ts

import WebSocket from "ws";
import { PrismaClient } from "@prisma/client";
import { logger } from "../utils/logger";
import { CallOrchestrator } from "../services/orchestrator";
import { callService } from "../services/callService";

const prisma = new PrismaClient();

export const handleAudioStream = (ws: WebSocket, req: any) => {
  let orchestrator: CallOrchestrator | null = null;
  let callSid: string | null = null;
  let conversationId: string | null = null;
  let startTime: Date | null = null;

  ws.on("message", async (message) => {
    try {
      const msg = JSON.parse(message.toString());

      switch (msg.event) {
        case "connected":
          logger.info("🔗 Audio Stream Connected");
          break;

        case "start":
          callSid = msg.start.callSid;
          const streamSid = msg.start.streamSid;
          const agentId = msg.start.customParameters?.agentId || "default";
          conversationId = msg.start.customParameters?.conversationId;
          startTime = new Date();
          
          logger.info(`📞 Call Started: ${callSid}, Agent: ${agentId}`);
          
          // Verify the conversation exists
          if (conversationId) {
            const conversation = await prisma.conversation.findUnique({
              where: { id: conversationId },
            });

            if (!conversation) {
              logger.error(`Conversation ${conversationId} not found`);
              ws.close(1008, 'Conversation not found');
              return;
            }
          }

          // Initialize the Orchestrator
          // This handles Ear, Brain, Mouth, and Interruption logic internally
          if (!callSid) {
            logger.error('No callSid provided - cannot initialize orchestrator');
            ws.close(1008, 'Missing callSid');
            return;
          }

          orchestrator = new CallOrchestrator(ws, callSid, streamSid, agentId);
          
          logger.info(`✅ Orchestrator initialized for call ${callSid}`);
          break;

        case "media":
          // Simply pass the raw audio to the Orchestrator
          if (orchestrator) {
            orchestrator.handleAudio(msg.media.payload);
          }
          break;

        case "stop":
          logger.info(`🛑 Call Stop Event: ${callSid}`);
          await handleCallEnd();
          break;
      }
    } catch (err) {
      logger.error({ err }, "WebSocket Message Error");
    }
  });

  ws.on("close", async () => {
    logger.info(`🔌 Stream Disconnected: ${callSid}`);
    await handleCallEnd();
  });

  ws.on("error", (error) => {
    logger.error({ error }, `WebSocket Error: ${callSid}`);
  });

  /**
   * Handle call end - cleanup and billing
   */
const handleCallEnd = async () => {
  try {
    // Cleanup orchestrator
    if (orchestrator) {
      orchestrator.cleanup();
      orchestrator = null;
    }

    // ✅ Add null checks at the beginning
    if (!callSid || !conversationId) {
      logger.warn('No callSid or conversation ID - skipping billing');
      return;
    }
      // Update conversation end time
      const conversation = await prisma.conversation.update({
        where: { id: conversationId },
        data: {
          end_time: new Date(),
          status: 'COMPLETED',
        },
        include: {
          agent: {
            include: { org: true },
          },
        },
      });

      logger.info(`📊 Call ended. Duration: ${getDurationMinutes(conversation)} minutes`);

      // ✅ BILLING: Deduct minutes and calculate cost
      try {
        const billingResult = await callService.handleCallEnd(conversationId);
        
        logger.info({
          conversationId,
          duration: billingResult.duration,
          cost: billingResult.cost,
          orgId: conversation.agent.org_id,
          remainingBalance: conversation.agent.org.credit_balance,
        }, '💰 Billing processed successfully');

        // Check if balance is getting low (< 100 minutes)
        if (conversation.agent.org.credit_balance < 100) {
          logger.warn({
            orgId: conversation.agent.org_id,
            balance: conversation.agent.org.credit_balance,
          }, '⚠️ Low balance warning - organization should be notified');
          
          // TODO: Send low balance notification email/webhook
        }

      } catch (billingError) {
        logger.error({ 
          error: billingError, 
          conversationId,
          orgId: conversation.agent.org_id,
        }, '❌ Billing processing failed');
        
        // Don't throw - we still want to clean up properly
        // But log this for manual review
      }

    } catch (error) {
      logger.error({ error, callSid, conversationId }, 'Error handling call end');
    }
  };

  /**
   * Helper to calculate duration in minutes
   */
  const getDurationMinutes = (conversation: any): number => {
    if (!conversation.start_time || !conversation.end_time) {
      return 0;
    }

    const durationMs = conversation.end_time.getTime() - conversation.start_time.getTime();
    return Math.ceil(durationMs / 60000); // Round up to nearest minute
  };
};