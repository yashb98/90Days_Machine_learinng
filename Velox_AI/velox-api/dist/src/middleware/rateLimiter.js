"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.rateLimiter = void 0;
const redis_1 = __importDefault(require("../config/redis"));
const logger_1 = require("../utils/logger");
const MAX_CALLS_PER_MINUTE = 50;
/**
 * Middleware to block Organizations exceeding limits.
 * Assumes req.user.org_id exists (set by Auth middleware later).
 */
const rateLimiter = async (req, res, next) => {
    // Mocking org_id for now (Replace this with actual Auth extraction later)
    const orgId = req.headers["x-org-id"] || "default_org";
    const key = `ratelimit:${orgId}:calls_per_min`;
    try {
        // Atomic increment
        const currentCount = await redis_1.default.incr(key);
        // If this is the first request, set the expiry window (60 seconds)
        if (currentCount === 1) {
            await redis_1.default.expire(key, 60);
        }
        if (currentCount > MAX_CALLS_PER_MINUTE) {
            logger_1.logger.warn(`🚫 Rate limit exceeded for Org ${orgId}`);
            res.status(429).json({
                error: "Too Many Requests",
                retry_after: await redis_1.default.ttl(key)
            });
            return;
        }
        // Add headers so the client knows their usage
        res.setHeader("X-RateLimit-Limit", MAX_CALLS_PER_MINUTE);
        res.setHeader("X-RateLimit-Remaining", MAX_CALLS_PER_MINUTE - currentCount);
        next();
    }
    catch (error) {
        logger_1.logger.error({ err: error }, "Rate limiter error");
        // Fail open: If Redis is down, let the request through rather than blocking everyone
        next();
    }
};
exports.rateLimiter = rateLimiter;
