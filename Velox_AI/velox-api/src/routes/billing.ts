// velox-api/src/routes/billing.ts

import { Router } from 'express';
import { billingService } from '../services/billingService';
import { logger } from '../utils/logger';

const router = Router();

// Create checkout session
router.post('/checkout', async (req, res) => {
  try {
    const { orgId, planType, successUrl, cancelUrl } = req.body;

    const session = await billingService.createCheckoutSession(
      orgId,
      planType,
      successUrl,
      cancelUrl
    );

    res.json({ sessionId: session.id, url: session.url });
  } catch (error: any) {
    logger.error({ error }, 'Failed to create checkout session');
    res.status(500).json({ error: error.message });
  }
});

// Get billing info
router.get('/:orgId', async (req, res) => {
  try {
    const { orgId } = req.params;
    const billingInfo = await billingService.getBillingInfo(orgId);
    res.json(billingInfo);
  } catch (error: any) {
    logger.error({ error }, 'Failed to get billing info');
    res.status(500).json({ error: error.message });
  }
});

// Cancel subscription
router.post('/:orgId/cancel', async (req, res) => {
  try {
    const { orgId } = req.params;
    await billingService.cancelSubscription(orgId);
    res.json({ success: true });
  } catch (error: any) {
    logger.error({ error }, 'Failed to cancel subscription');
    res.status(500).json({ error: error.message });
  }
});

export default router;