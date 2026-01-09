// src/db.ts
import { Pool } from "pg";
import "dotenv/config";
import pino from "pino";

// Create a standalone logger to avoid circular dependency with app.ts
const logger = pino({
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

export const pool = new Pool({
  connectionString,
  // Add connection error handling
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// Test connection on startup
pool.on("error", (err) => {
  logger.error({ error: err.message }, "Unexpected database pool error");
});

// Optional: Test connection (don't block startup if it fails)
pool.query("SELECT NOW()")
  .then(() => {
    logger.info("✅ Database connection successful");
  })
  .catch((err) => {
    logger.error({ 
      error: err.message,
      hint: "Make sure PostgreSQL is running and DATABASE_URL is correct"
    }, "❌ Database connection failed");
  });
