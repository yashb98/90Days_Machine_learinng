import { Request, Response, NextFunction } from "express";
import redis from "../config/redis";
import { logger } from "../utils/logger";

const MAX_CALLS_PER_MINUTE = 50;

/**
 * Middleware to block Organizations exceeding limits.
 * Assumes req.user.org_id exists (set by Auth middleware later).
 */
export const rateLimiter = async (req: Request, res: Response, next: NextFunction) => {
  // Mocking org_id for now (Replace this with actual Auth extraction later)
  const orgId = req.headers["x-org-id"] || "default_org"; 
  
  const key = `ratelimit:${orgId}:calls_per_min`;

  try {
    // Atomic increment
    const currentCount = await redis.incr(key);

    // If this is the first request, set the expiry window (60 seconds)
    if (currentCount === 1) {
      await redis.expire(key, 60);
    }

    if (currentCount > MAX_CALLS_PER_MINUTE) {
      logger.warn(`🚫 Rate limit exceeded for Org ${orgId}`);
      res.status(429).json({ 
        error: "Too Many Requests", 
        retry_after: await redis.ttl(key) 
      });
      return;
    }

    // Add headers so the client knows their usage
    res.setHeader("X-RateLimit-Limit", MAX_CALLS_PER_MINUTE);
    res.setHeader("X-RateLimit-Remaining", MAX_CALLS_PER_MINUTE - currentCount);

    next();
  } catch (error) {
    logger.error({err: error}, "Rate limiter error");
    // Fail open: If Redis is down, let the request through rather than blocking everyone
    next();
  }
};