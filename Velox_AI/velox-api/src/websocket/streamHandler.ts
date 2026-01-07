import WebSocket from "ws";
import { logger } from "../app";
import { SessionService } from "../services/sessionService";
import { TranscriptionService } from "../services/transcriptionService";
import { LLMService } from "../services/llmService";
import { TtsService } from "../services/ttsService"; // <--- 1. Import TTS

export const handleAudioStream = (ws: WebSocket, req: any) => {
  let streamSid = "";
  let callSid = "";
  
  // Services
  let transcriptionService: TranscriptionService | null = null;
  let llmService: LLMService | null = null;
  let ttsService: TtsService | null = null;

  //  Gatekeeper ID: Increments on every new user turn/interruption
  let currentInteractionId = 0;

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
          
          await SessionService.initSession(callSid, agentId);

          llmService = new LLMService();
          ttsService = new TtsService();

          transcriptionService = new TranscriptionService(
            // Callback 1: User Finished Speaking (Transcript Ready)
            async (userText) => {
              // Start a new turn
              currentInteractionId++;
              const myId = currentInteractionId; // Lock ID for this turn

              if (llmService && ttsService) {
                await llmService.generateResponse(userText, async (aiSentence) => {
                  
                  // ⚡ Race Condition Check 1: Did user interrupt logic?
                  if (myId !== currentInteractionId) return;

                  logger.info(`🗣️ SPEAKING: ${aiSentence}`);
                  const audioBuffer = await ttsService!.generateAudio(aiSentence);

                  // ⚡ Race Condition Check 2: Did user interrupt TTS?
                  if (myId !== currentInteractionId) return;

                  if (audioBuffer && streamSid) {
                    const mediaMessage = {
                      event: "media",
                      streamSid: streamSid,
                      media: { payload: audioBuffer.toString("base64") },
                    };
                    ws.send(JSON.stringify(mediaMessage));
                  }
                });
              }
            },

            // Callback 2: Interruption Detected (Barge-In)
            () => {
              // 1. Invalidate any pending AI audio
              currentInteractionId++;
              
              // 2. Clear Twilio's Audio Buffer immediately
              if (streamSid) {
                logger.info(" Interruption! Clearing Twilio Buffer.");
                const clearMessage = {
                  event: "clear",
                  streamSid: streamSid
                };
                ws.send(JSON.stringify(clearMessage));
              }
            }
          );
          // --- START TEST CODE ---
          setTimeout(() => {
            logger.info("⚡ SIMULATING: User Interruption Triggered!");
            
            // 1. Manually trigger the interruption logic
            currentInteractionId++; 
            if (streamSid) {
              logger.info("🛑 Interruption! Clearing Twilio Buffer.");
              const clearMessage = { event: "clear", streamSid: streamSid };
              ws.send(JSON.stringify(clearMessage));
            }
          }, 3000); // Trigger after 3 seconds
          // --- END TEST CODE ---
          break;

        case "media":
          if (transcriptionService) {
            transcriptionService.send(msg.media.payload);
          }
          break;

        case "stop":
          logger.info(` Call Ended: ${callSid}`);
          if (transcriptionService) transcriptionService.close();
          break;
      }
    } catch (err) {
      logger.error({ err }, "WebSocket Message Error");
    }
  });

  ws.on("close", () => {
    logger.info(" Stream Disconnected");
    if (transcriptionService) transcriptionService.close();
  });
};