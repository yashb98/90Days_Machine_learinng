"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.app = void 0;
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const helmet_1 = __importDefault(require("helmet"));
const pino_http_1 = require("pino-http");
const rateLimiter_1 = require("./middleware/rateLimiter");
// 1. IMPORT THE LOGGER FROM UTILS (Fixes Circular Dependency)
const logger_1 = require("./utils/logger");
const documentRoutes_1 = __importDefault(require("./routes/documentRoutes"));
const playground_1 = __importDefault(require("./routes/playground"));
// UUID Setup (Keep your existing logic)
let uuidv4;
import("uuid").then((uuid) => {
    uuidv4 = uuid.v4;
});
const app = (0, express_1.default)();
exports.app = app;
// 2. Security & Parsing
app.use((0, helmet_1.default)());
app.use((0, cors_1.default)());
app.use(express_1.default.json());
app.use(rateLimiter_1.rateLimiter);
// 3. Request ID Middleware (The "Trace")
app.use((req, res, next) => {
    req.id = req.headers["x-request-id"] || (uuidv4 ? uuidv4() : "init-id");
    res.setHeader("X-Request-ID", req.id);
    next();
});
// 4. Logger Middleware
// Uses the shared 'logger' from utils, so no more crashes!
app.use((0, pino_http_1.pinoHttp)({
    logger: logger_1.logger,
    genReqId: (req) => req.id,
}));
// 5. Health Check
app.get("/health", (req, res) => {
    res.status(200).json({ status: "ok", version: process.env.npm_package_version });
});
// 6. Routes
app.use("/api/documents", documentRoutes_1.default);
app.use('/api/playground', playground_1.default);
