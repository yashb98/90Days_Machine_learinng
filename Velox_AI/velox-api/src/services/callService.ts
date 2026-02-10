// velox-api/src/services/callService.ts

import { PrismaClient } from '@prisma/client';
import { billingService } from './billingService';
import { logger } from '../utils/logger';

const prisma = new PrismaClient();

export class CallService {
  /**
   * Handle call end - calculate duration and deduct minutes
   */
  async handleCallEnd(conversationId: string) {
    try {
      // Get conversation with start/end times
      const conversation = await prisma.conversation.findUnique({
        where: { id: conversationId },
        include: {
          agent: {
            include: { org: true },
          },
        },
      });

      if (!conversation || !conversation.end_time) {
        throw new Error('Invalid conversation');
      }

      // Calculate duration in minutes (rounded up)
      const durationMs = conversation.end_time.getTime() - conversation.start_time.getTime();
      const durationMinutes = Math.ceil(durationMs / 60000);

      logger.info(`Call ended. Duration: ${durationMinutes} minutes`);

      // Deduct minutes from organization
      const success = await billingService.deductMinutes(
        conversation.agent.org_id,
        durationMinutes,
        conversationId
      );

      if (!success) {
        logger.error(`Failed to deduct ${durationMinutes} minutes from org ${conversation.agent.org_id}`);
        
        // Send alert email/notification about insufficient balance
        // This shouldn't happen if enforcement is working, but handle gracefully
      }

      // Update conversation with cost
      const costPerMinute = 0.05; // $0.05 per minute
      const totalCost = durationMinutes * costPerMinute;

      await prisma.conversation.update({
        where: { id: conversationId },
        data: {
          cost_accrued: totalCost,
        },
      });

      logger.info(`Updated conversation ${conversationId} with cost: $${totalCost.toFixed(2)}`);

      return {
        duration: durationMinutes,
        cost: totalCost,
      };
    } catch (error) {
      logger.error({ error }, 'Failed to handle call end');
      throw error;
    }
  }
}

export const callService = new CallService();