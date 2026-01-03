import WebSocket from "ws";
import { logger } from "../app";
import { SessionService } from "../services/sessionService";

export const handleAudioStream = (ws: WebSocket, req: any) => {
  logger.info("🔌 New Twilio Stream Connected");
  let streamSid = "";
  let callSid = "";

  ws.on("message", async (message) => {
    try {
      const msg = JSON.parse(message.toString());

      switch (msg.event) {
        case "connected":
          logger.info("Audio Stream Connected");
          break;

        case "start":
          streamSid = msg.start.streamSid;
          callSid = msg.start.callSid;
          const agentId = msg.start.customParameters?.agentId || "unknown";
          
          logger.info(`Call Started: ${callSid} | Stream: ${streamSid}`);
          
          // Initialize Redis State from Day 4
          await SessionService.initSession(callSid, agentId);
          break;

        case "media":
          // This is the raw audio (base64 encoded u-law)
          // For now, we just acknowledge we received it.
          // In Day 6-7, we send this to Deepgram/OpenAI.
          const payload = msg.media.payload;
          const chunk = msg.media.chunk;
          if (parseInt(chunk) % 50 === 0) { 
             // Log every 50th packet so we don't spam console
             process.stdout.write("."); 
          }
          break;

        case "stop":
          logger.info(`Call Ended: ${callSid}`);
          break;
      }
    } catch (err) {
      logger.error({ err }, "WebSocket Message Error");
    }
  });

  ws.on("close", () => {
    logger.info("🔌 Stream Disconnected");
  });
};