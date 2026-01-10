"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
// velox-api/src/server.ts
require("dotenv/config");
const app_1 = require("./app");
const logger_1 = require("./utils/logger");
const http_1 = require("http");
const ws_1 = require("ws");
const voice_1 = __importDefault(require("./routes/voice"));
const streamHandler_1 = require("./websocket/streamHandler");
const PORT = process.env.PORT || 8080; // Cloud Run defaults to 8080
app_1.app.use("/voice", voice_1.default);
// Create HTTP server
const server = (0, http_1.createServer)(app_1.app);
// Create WebSocket server
const wss = new ws_1.WebSocketServer({ server, path: "/streams" });
wss.on("connection", (ws, req) => {
    logger_1.logger.info("New WebSocket connection established");
    (0, streamHandler_1.handleAudioStream)(ws, req);
});
server.listen(PORT, () => {
    logger_1.logger.info(` Server listening on port ${PORT}`);
});
// Graceful Shutdown (Handle Cloud Run SIGTERM)
process.on("SIGTERM", () => {
    logger_1.logger.info("SIGTERM received. Shutting down gracefully...");
    server.close(() => {
        logger_1.logger.info("Server closed.");
        process.exit(0);
    });
});
