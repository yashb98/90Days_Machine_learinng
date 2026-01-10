"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CallOrchestrator = void 0;
const ws_1 = __importDefault(require("ws"));
const logger_1 = require("../utils/logger");
const transcriptionService_1 = require("./transcriptionService");
const llmService_1 = require("./llmService");
const ttsService_1 = require("./ttsService");
const sessionService_1 = require("./sessionService");
const metricsService_1 = require("./metricsService");
const retrievalService_1 = require("./retrievalService");
class CallOrchestrator {
    constructor(ws, callSid, streamSid, agentId) {
        // Services
        this.transcriptionService = null;
        // State
        this.currentInteractionId = 0;
        this.isAlive = true;
        this.ws = ws;
        this.callSid = callSid;
        this.streamSid = streamSid;
        this.agentId = agentId;
        this.metrics = new metricsService_1.MetricsService();
        this.retrievalService = new retrievalService_1.RetrievalService();
        // Initialize Static Services
        this.llmService = new llmService_1.LLMService();
        this.ttsService = new ttsService_1.TtsService();
        // Initialize Session
        sessionService_1.SessionService.initSession(this.callSid, this.agentId);
        this.setupPipeline();
    }
    setupPipeline() {
        this.transcriptionService = new transcriptionService_1.TranscriptionService(
        // 1. User Finished Speaking
        async (text) => this.handleUserMessage(text), 
        // 2. User Interrupted
        () => this.handleInterruption());
    }
    /**
     * Core Logic Loop: Ear -> Brain -> Mouth
     */
    async handleUserMessage(userText) {
        if (!this.isAlive)
            return;
        this.currentInteractionId++;
        const myId = this.currentInteractionId;
        // ⏱️ START TIMER (User stopped speaking)
        this.metrics.startTurn(myId);
        try {
            // Search the database for relevant info before asking LLM
            const context = await this.retrievalService.search(userText);
            if (context) {
                logger_1.logger.info(`🔍 Found ${context.length} relevant chunks in the database.`);
            }
            else {
                logger_1.logger.info("🔍 No relevant chunks found in the database.");
            }
            // ⏱️ Mark LLM Start
            this.metrics.mark(myId, "llmStart");
            await this.llmService.generateResponse(userText, async (aiSentence) => {
                if (myId !== this.currentInteractionId)
                    return;
                // ⏱️ Mark LLM First Token (We got the first sentence)
                this.metrics.mark(myId, "llmFirstToken");
                // ⏱️ Mark TTS Start
                this.metrics.mark(myId, "ttsStart");
                const audio = await this.ttsService.generateAudio(aiSentence);
                // ⏱️ Mark TTS First Byte (We got audio)
                this.metrics.mark(myId, "ttsFirstByte");
                if (myId !== this.currentInteractionId)
                    return;
                if (audio)
                    this.sendAudio(audio);
            });
        }
        catch (error) {
            logger_1.logger.error({ error }, "Pipeline Error");
            this.playFallbackError();
        }
    }
    /**
     * Handle Barge-In
     */
    handleInterruption() {
        logger_1.logger.info(" Interruption detected");
        this.currentInteractionId++; // Invalidate pending actions
        this.sendClearMessage();
    }
    /**
     * Error Recovery: The "Safety Net"
     */
    async playFallbackError() {
        logger_1.logger.warn(" Triggering Fallback Audio");
        // In a real app, load a pre-recorded WAV file here.
        // For now, we try to generate a quick apology.
        try {
            const audio = await this.ttsService.generateAudio("I'm having trouble connecting. One moment.");
            if (audio)
                this.sendAudio(audio);
        }
        catch (e) {
            logger_1.logger.error(" Critical: Even Fallback Failed");
        }
    }
    // --- WebSocket Helpers ---
    handleAudio(payload) {
        if (this.transcriptionService) {
            this.transcriptionService.send(payload);
        }
    }
    sendAudio(audio) {
        const mediaMessage = {
            event: "media",
            streamSid: this.streamSid,
            media: { payload: audio.toString("base64") },
        };
        if (this.ws.readyState === ws_1.default.OPEN) {
            this.ws.send(JSON.stringify(mediaMessage));
        }
    }
    sendClearMessage() {
        const clearMessage = { event: "clear", streamSid: this.streamSid };
        if (this.ws.readyState === ws_1.default.OPEN) {
            this.ws.send(JSON.stringify(clearMessage));
        }
    }
    cleanup() {
        this.isAlive = false;
        if (this.transcriptionService)
            this.transcriptionService.close();
        logger_1.logger.info(` Orchestrator cleaned up for ${this.callSid}`);
    }
}
exports.CallOrchestrator = CallOrchestrator;
