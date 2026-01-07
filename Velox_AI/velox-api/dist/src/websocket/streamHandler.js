"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.handleAudioStream = void 0;
const app_1 = require("../app");
const sessionService_1 = require("../services/sessionService");
const transcriptionService_1 = require("../services/transcriptionService");
const llmService_1 = require("../services/llmService");
const ttsService_1 = require("../services/ttsService"); // <--- 1. Import TTS
const handleAudioStream = (ws, req) => {
    let streamSid = "";
    let callSid = "";
    // Services
    let transcriptionService = null;
    let llmService = null;
    let ttsService = null;
    //  Gatekeeper ID: Increments on every new user turn/interruption
    let currentInteractionId = 0;
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
                    await sessionService_1.SessionService.initSession(callSid, agentId);
                    llmService = new llmService_1.LLMService();
                    ttsService = new ttsService_1.TtsService();
                    transcriptionService = new transcriptionService_1.TranscriptionService(
                    // Callback 1: User Finished Speaking (Transcript Ready)
                    async (userText) => {
                        // Start a new turn
                        currentInteractionId++;
                        const myId = currentInteractionId; // Lock ID for this turn
                        if (llmService && ttsService) {
                            await llmService.generateResponse(userText, async (aiSentence) => {
                                // ⚡ Race Condition Check 1: Did user interrupt logic?
                                if (myId !== currentInteractionId)
                                    return;
                                app_1.logger.info(`🗣️ SPEAKING: ${aiSentence}`);
                                const audioBuffer = await ttsService.generateAudio(aiSentence);
                                // ⚡ Race Condition Check 2: Did user interrupt TTS?
                                if (myId !== currentInteractionId)
                                    return;
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
                            app_1.logger.info(" Interruption! Clearing Twilio Buffer.");
                            const clearMessage = {
                                event: "clear",
                                streamSid: streamSid
                            };
                            ws.send(JSON.stringify(clearMessage));
                        }
                    });
                    // --- START TEST CODE ---
                    setTimeout(() => {
                        app_1.logger.info("⚡ SIMULATING: User Interruption Triggered!");
                        // 1. Manually trigger the interruption logic
                        currentInteractionId++;
                        if (streamSid) {
                            app_1.logger.info("🛑 Interruption! Clearing Twilio Buffer.");
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
                    app_1.logger.info(` Call Ended: ${callSid}`);
                    if (transcriptionService)
                        transcriptionService.close();
                    break;
            }
        }
        catch (err) {
            app_1.logger.error({ err }, "WebSocket Message Error");
        }
    });
    ws.on("close", () => {
        app_1.logger.info(" Stream Disconnected");
        if (transcriptionService)
            transcriptionService.close();
    });
};
exports.handleAudioStream = handleAudioStream;
