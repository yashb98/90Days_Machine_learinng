"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.handleAudioStream = void 0;
const logger_1 = require("../utils/logger");
const orchestrator_1 = require("../services/orchestrator"); // The new Manager
const handleAudioStream = (ws, req) => {
    // We no longer manage individual services here. 
    // We just hold one instance of the Orchestrator.
    let orchestrator = null;
    ws.on("message", (message) => {
        try {
            const msg = JSON.parse(message.toString());
            switch (msg.event) {
                case "connected":
                    logger_1.logger.info("Audio Stream Connected");
                    break;
                case "start":
                    const { callSid, streamSid } = msg.start;
                    const agentId = msg.start.customParameters?.agentId || "default";
                    logger_1.logger.info(`📞 Call Started: ${callSid}`);
                    // Initialize the Orchestrator
                    // This handles Ear, Brain, Mouth, and Interruption logic internally
                    orchestrator = new orchestrator_1.CallOrchestrator(ws, callSid, streamSid, agentId);
                    break;
                case "media":
                    // Simply pass the raw audio to the Orchestrator
                    if (orchestrator) {
                        orchestrator.handleAudio(msg.media.payload);
                    }
                    break;
                case "stop":
                    logger_1.logger.info("Call Ended");
                    if (orchestrator)
                        orchestrator.cleanup();
                    break;
            }
        }
        catch (err) {
            logger_1.logger.error({ err }, "WebSocket Message Error");
        }
    });
    ws.on("close", () => {
        logger_1.logger.info("🔌 Stream Disconnected");
        if (orchestrator)
            orchestrator.cleanup();
    });
};
exports.handleAudioStream = handleAudioStream;
