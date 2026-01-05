import WebSocket from "ws";
import { logger } from "../app";
import { SessionService } from "../services/sessionService";
import { TranscriptionService } from "../services/transcriptionService";
import { LLMService } from "../services/llmService";

export const handleAudioStream = (ws: WebSocket, req: any) => {
  let streamSid = "";
  let callSid = "";
  let transcriptionService: TranscriptionService | null = null;
  let llmService: LLMService | null = null; // <--- The Brain

  ws.on("message", async (message) => {
    try {
      const msg = JSON.parse(message.toString());

      switch (msg.event) {
        case "connected":
          logger.info(" Audio Stream Connected");
          break;

        case "start":
          streamSid = msg.start.streamSid;
          callSid = msg.start.callSid;
          const agentId = msg.start.customParameters?.agentId || "unknown";
          
          logger.info(`📞 Call Started: ${callSid}`);
          
          // 1. Initialize Redis Session
          await SessionService.initSession(callSid, agentId);

          // 2. Initialize the Brain (LLM)
          llmService = new LLMService();

          // 3. Initialize the Ear (Deepgram) with a Callback
          // When Deepgram hears a full sentence, it calls this function:
          transcriptionService = new TranscriptionService(async (userText) => {
            if (llmService) {
              // Send the text to the Brain
              await llmService.generateResponse(userText, (aiSentence) => {
                 // The Brain returned a full sentence
                 logger.info(`🗣️ SPEAKING: ${aiSentence}`);
                 // TODO (Day 8): Send this text to ElevenLabs TTS
              });
            }
          });
          break;

        case "media":
          // Pipe Audio to Deepgram
          if (transcriptionService) {
            transcriptionService.send(msg.media.payload);
          }
          break;

        case "stop":
          logger.info(`🛑 Call Ended: ${callSid}`);
          if (transcriptionService) {
            transcriptionService.close();
          }
          break;
      }
    } catch (err) {
      logger.error({ err }, "WebSocket Message Error");
    }
  });

  ws.on("close", () => {
    logger.info("🔌 Stream Disconnected");
    if (transcriptionService) {
      transcriptionService.close();
    }
  });
};