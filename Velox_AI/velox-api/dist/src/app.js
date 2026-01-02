"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.logger = exports.app = void 0;
// velox-api/src/app.ts
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const helmet_1 = __importDefault(require("helmet"));
const pino_http_1 = require("pino-http");
const pino_1 = __importDefault(require("pino"));
const rateLimiter_1 = require("./middleware/rateLimiter");
let uuidv4;
import("uuid").then((uuid) => {
    uuidv4 = uuid.v4;
});
const logger = (0, pino_1.default)({
    level: process.env.LOG_LEVEL || "info",
    // In local dev, print pretty logs. In prod, print JSON.
    transport: process.env.NODE_ENV !== "production" ? { target: "pino-pretty" } : undefined,
});
exports.logger = logger;
const app = (0, express_1.default)();
exports.app = app;
// 1. Security & Parsing
app.use((0, helmet_1.default)());
app.use((0, cors_1.default)());
app.use(express_1.default.json());
app.use(rateLimiter_1.rateLimiter);
// 2. Request ID Middleware (The "Trace")
app.use((req, res, next) => {
    req.id = req.headers["x-request-id"] || uuidv4();
    res.setHeader("X-Request-ID", req.id);
    next();
});
// 3. Logger (Auto-logs every request with latency & ID)
app.use((0, pino_http_1.pinoHttp)({
    logger,
    genReqId: (req) => req.id, // Link logger to the Request ID
}));
// 4. Health Check (Critical for Cloud Run)
app.get("/health", (req, res) => {
    res.status(200).json({ status: "ok", version: process.env.npm_package_version });
});
