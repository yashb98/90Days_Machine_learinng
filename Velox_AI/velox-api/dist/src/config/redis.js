"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const ioredis_1 = __importDefault(require("ioredis"));
const logger_1 = require("../utils/logger");
const redisConfig = {
    host: process.env.REDIS_HOST || "localhost",
    port: parseInt(process.env.REDIS_PORT || "6379"),
    // Fail fast if Redis is down so the app knows immediately
    retryStrategy: (times) => {
        const delay = Math.min(times * 50, 2000);
        return delay;
    },
};
// The Singleton Instance
const redis = new ioredis_1.default(redisConfig);
redis.on("connect", () => {
    logger_1.logger.info("Redis connected successfully");
});
redis.on("error", (err) => {
    logger_1.logger.error({ err }, "Redis connection error");
});
exports.default = redis;
