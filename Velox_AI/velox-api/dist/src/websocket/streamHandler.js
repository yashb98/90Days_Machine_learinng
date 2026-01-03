"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.handleAudioStream = void 0;
const app_1 = require("../app");
const sessionService_1 = require("../services/sessionService");
const handleAudioStream = (ws, req) => {
    app_1.logger.info("🔌 New Twilio Stream Connected");
    let streamSid = "";
    let callSid = "";
    ws.on("message", async (message) => {
        try {
            const msg = JSON.parse(message.toString());
            switch (msg.event) {
                case "connected":
                    app_1.logger.info("Audio Stream Connected");
                    break;
                case "start":
                    streamSid = msg.start.streamSid;
                    callSid = msg.start.callSid;
                    const agentId = msg.start.customParameters?.agentId || "unknown";
                    app_1.logger.info(`Call Started: ${callSid} | Stream: ${streamSid}`);
                    // Initialize Redis State from Day 4
                    await sessionService_1.SessionService.initSession(callSid, agentId);
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
                    app_1.logger.info(`Call Ended: ${callSid}`);
                    break;
            }
        }
        catch (err) {
            app_1.logger.error({ err }, "WebSocket Message Error");
        }
    });
    ws.on("close", () => {
        app_1.logger.info("🔌 Stream Disconnected");
    });
};
exports.handleAudioStream = handleAudioStream;
