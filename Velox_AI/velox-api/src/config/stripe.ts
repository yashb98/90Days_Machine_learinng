// velox-api/src/config/stripe.ts

import Stripe from 'stripe';
import { logger } from '../utils/logger';

if (!process.env.STRIPE_SECRET_KEY) {
  throw new Error('STRIPE_SECRET_KEY is required');
}

export const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
  // apiVersion: '2024-12-18.acacia',
});

// Product and Price IDs (you'll create these in Stripe Dashboard)
export const STRIPE_PLANS = {
  STARTER: {
    name: 'Starter',
    price: 4900, // $49.00 in cents
    minutes: 1000,
    priceId: process.env.STRIPE_STARTER_PRICE_ID || '',
  },
  PRO: {
    name: 'Pro',
    price: 19900, // $199.00 in cents
    minutes: 5000,
    priceId: process.env.STRIPE_PRO_PRICE_ID || '',
  },
  ENTERPRISE: {
    name: 'Enterprise',
    price: 49900, // $499.00 in cents
    minutes: 20000,
    priceId: process.env.STRIPE_ENTERPRISE_PRICE_ID || '',
  },
};

logger.info('Stripe initialized');