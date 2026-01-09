// velox-api/src/app.ts
import express from "express";
import cors from "cors";
import helmet from "helmet";
import { pinoHttp } from "pino-http";
import pino from "pino";
import { rateLimiter } from "./middleware/rateLimiter";
import type { Redis } from "ioredis";
import documentRoutes from "./routes/documentRoutes";

let uuidv4: () => string;

import("uuid").then((uuid) => {
  uuidv4 = uuid.v4;
});

const logger = pino({
  level: process.env.LOG_LEVEL || "info",
  // In local dev, print pretty logs. In prod, print JSON.
  transport: process.env.NODE_ENV !== "production" ? { target: "pino-pretty" } : undefined,
});

const app = express();

// 1. Security & Parsing
app.use(helmet());
app.use(cors());
app.use(express.json());

app.use(rateLimiter);

// 2. Request ID Middleware (The "Trace")
app.use((req, res, next) => {
  req.id = req.headers["x-request-id"] as string || uuidv4();
  res.setHeader("X-Request-ID", req.id);
  next();
});

// 3. Logger (Auto-logs every request with latency & ID)
app.use(pinoHttp({ 
  logger,
  genReqId: (req) => req.id, // Link logger to the Request ID
}));

// 4. Health Check (Critical for Cloud Run)
app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok", version: process.env.npm_package_version });
});

app.use("/api/documents", documentRoutes);
// Placeholder for Routes (We'll add these later)
// app.use("/api/v1", routes);

export { app, logger };