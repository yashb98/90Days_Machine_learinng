// velox-api/src/server.ts
import "dotenv/config";
import { app } from "./app";
import { logger} from "./utils/logger";
import {createServer} from "http";
import {WebSocketServer} from "ws";
import voiceRoutes from "./routes/voice";
import {handleAudioStream} from "./websocket/streamHandler";

const PORT = process.env.PORT || 8080; // Cloud Run defaults to 8080


app.use("/voice", voiceRoutes);

// Create HTTP server
const server = createServer(app);

// Create WebSocket server
const wss = new WebSocketServer({ server, path: "/media-stream" });

wss.on("connection", (ws, req) => {
  logger.info("New WebSocket connection established");
  handleAudioStream(ws, req);
});

server.listen(PORT, () => {
  logger.info(` Server listening on port ${PORT}`);
});

// Graceful Shutdown (Handle Cloud Run SIGTERM)
process.on("SIGTERM", () => {
  logger.info("SIGTERM received. Shutting down gracefully...");
  server.close(() => {
    logger.info("Server closed.");
    process.exit(0);
  });
});