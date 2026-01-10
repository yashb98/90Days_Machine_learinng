import WebSocket from "ws";
import { logger } from "../utils/logger";
import { CallOrchestrator } from "../services/orchestrator"; // The new Manager
import { RetrievalService } from "../services/retrievalService";

export const handleAudioStream = (ws: WebSocket, req: any) => {
  // We no longer manage individual services here. 
  // We just hold one instance of the Orchestrator.
  let orchestrator: CallOrchestrator | null = null;

  ws.on("message", (message) => {
    try {
      const msg = JSON.parse(message.toString());

      switch (msg.event) {
        case "connected":
          logger.info("Audio Stream Connected");
          break;

        case "start":
          const { callSid, streamSid } = msg.start;
          const agentId = msg.start.customParameters?.agentId || "default";
          
          logger.info(`📞 Call Started: ${callSid}`);
          
          // Initialize the Orchestrator
          // This handles Ear, Brain, Mouth, and Interruption logic internally
          orchestrator = new CallOrchestrator(ws, callSid, streamSid, agentId);
          break;

        case "media":
          // Simply pass the raw audio to the Orchestrator
          if (orchestrator) {
            orchestrator.handleAudio(msg.media.payload);
          }
          break;

        case "stop":
          logger.info("Call Ended");
          if (orchestrator) orchestrator.cleanup();
          break;
      }
    } catch (err) {
      logger.error({ err }, "WebSocket Message Error");
    }
  });

  ws.on("close", () => {
    logger.info("🔌 Stream Disconnected");
    if (orchestrator) orchestrator.cleanup();
  });
};