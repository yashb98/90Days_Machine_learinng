import express from "express";
import cors from "cors";
import helmet from "helmet";
import { pinoHttp } from "pino-http";
import { rateLimiter } from "./middleware/rateLimiter";
// 1. IMPORT THE LOGGER FROM UTILS (Fixes Circular Dependency)
import { logger } from "./utils/logger"; 
import documentRoutes from "./routes/documentRoutes";
import playgroundRoutes from './routes/playground';
import billingRoutes from './routes/billing';

// UUID Setup (Keep your existing logic)
let uuidv4: () => string;
import("uuid").then((uuid) => {
  uuidv4 = uuid.v4;
});

const app = express();

// 2. Security & Parsing
app.use(helmet());
app.use(cors());
app.use(express.json());

app.use(rateLimiter);

// 3. Request ID Middleware (The "Trace")
app.use((req, res, next) => {
  req.id = req.headers["x-request-id"] as string || (uuidv4 ? uuidv4() : "init-id");
  res.setHeader("X-Request-ID", req.id);
  next();
});

// 4. Logger Middleware
// Uses the shared 'logger' from utils, so no more crashes!
app.use(pinoHttp({ 
  logger,
  genReqId: (req) => req.id, 
}));

// 5. Health Check
app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok", version: process.env.npm_package_version });
});

// 6. Routes
app.use("/api/documents", documentRoutes);
app.use('/api/playground', playgroundRoutes);
app.use('/api/billing', billingRoutes);

export { app };