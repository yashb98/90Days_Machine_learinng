import Redis from "ioredis";
import { logger } from "../utils/logger";

const redisConfig = {
  host: process.env.REDIS_HOST || "localhost",
  port: parseInt(process.env.REDIS_PORT || "6379"),
  // Fail fast if Redis is down so the app knows immediately
  retryStrategy: (times: number) => {
    const delay = Math.min(times * 50, 2000);
    return delay;
  },
};

// The Singleton Instance
const redis = new Redis(redisConfig);

redis.on("connect", () => {
  logger.info("Redis connected successfully");
});

redis.on("error", (err) => {
  logger.error({ err }, "Redis connection error");
});

export default redis;