"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.handleAudioStream = void 0;
const app_1 = require("../app");
const sessionService_1 = require("../services/sessionService");
const transcriptionService_1 = require("../services/transcriptionService");
const llmService_1 = require("../services/llmService");
const handleAudioStream = (ws, req) => {
    let streamSid = "";
    let callSid = "";
    let transcriptionService = null;
    let llmService = null; // <--- The Brain
    ws.on("message", async (message) => {
        try {
            const msg = JSON.parse(message.toString());
            switch (msg.event) {
                case "connected":
                    app_1.logger.info(" Audio Stream Connected");
                    break;
                case "start":
                    streamSid = msg.start.streamSid;
                    callSid = msg.start.callSid;
                    const agentId = msg.start.customParameters?.agentId || "unknown";
                    app_1.logger.info(`📞 Call Started: ${callSid}`);
                    // 1. Initialize Redis Session
                    await sessionService_1.SessionService.initSession(callSid, agentId);
                    // 2. Initialize the Brain (LLM)
                    llmService = new llmService_1.LLMService();
                    // 3. Initialize the Ear (Deepgram) with a Callback
                    // When Deepgram hears a full sentence, it calls this function:
                    transcriptionService = new transcriptionService_1.TranscriptionService(async (userText) => {
                        if (llmService) {
                            // Send the text to the Brain
                            await llmService.generateResponse(userText, (aiSentence) => {
                                // The Brain returned a full sentence
                                app_1.logger.info(`🗣️ SPEAKING: ${aiSentence}`);
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
                    app_1.logger.info(`🛑 Call Ended: ${callSid}`);
                    if (transcriptionService) {
                        transcriptionService.close();
                    }
                    break;
            }
        }
        catch (err) {
            app_1.logger.error({ err }, "WebSocket Message Error");
        }
    });
    ws.on("close", () => {
        app_1.logger.info("🔌 Stream Disconnected");
        if (transcriptionService) {
            transcriptionService.close();
        }
    });
};
exports.handleAudioStream = handleAudioStream;
