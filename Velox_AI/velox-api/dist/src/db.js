"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.pool = void 0;
// src/db.ts
const pg_1 = require("pg");
require("dotenv/config");
const pino_1 = __importDefault(require("pino"));
// Create a standalone logger to avoid circular dependency with app.ts
const logger = (0, pino_1.default)({
    level: process.env.LOG_LEVEL || "info",
    transport: process.env.NODE_ENV !== "production" ? { target: "pino-pretty" } : undefined,
});
const connectionString = process.env.DATABASE_URL;
if (!connectionString) {
    logger.error("⚠️ DATABASE_URL is not set in environment variables");
    logger.info("Expected format: postgresql://user:password@host:port/database");
    logger.info("Example: postgresql://postgres:devpass@localhost:5432/velox_local");
}
// Mask password in logs for security
const maskedUrl = connectionString
    ? connectionString.replace(/:([^:@]+)@/, ":****@")
    : "not set";
logger.info(`📊 Database connection string: ${maskedUrl}`);
exports.pool = new pg_1.Pool({
    connectionString,
    // Add connection error handling
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
});
// Test connection on startup
exports.pool.on("error", (err) => {
    logger.error({ error: err.message }, "Unexpected database pool error");
});
// Optional: Test connection (don't block startup if it fails)
exports.pool.query("SELECT NOW()")
    .then(() => {
    logger.info("✅ Database connection successful");
})
    .catch((err) => {
    logger.error({
        error: err.message,
        hint: "Make sure PostgreSQL is running and DATABASE_URL is correct"
    }, "❌ Database connection failed");
});
