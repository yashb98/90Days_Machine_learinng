import WebSocket from "ws";
import { logger } from "../utils/logger";
import { TranscriptionService } from "./transcriptionService";
import { LLMService } from "./llmService";
import { TtsService } from "./ttsService";
import { SessionService } from "./sessionService";
import { MetricsService } from "./metricsService";
import { RetrievalService } from "./retrievalService";


export class CallOrchestrator {
  private ws: WebSocket;
  private callSid: string;
  private streamSid: string;
  private agentId: string;
  private metrics: MetricsService;
  private retrievalService: RetrievalService;
  // Services
  private transcriptionService: TranscriptionService | null = null;
  private llmService: LLMService;
  private ttsService: TtsService;
  
  // State
  private currentInteractionId = 0;
  private isAlive = true;

  constructor(ws: WebSocket, callSid: string, streamSid: string, agentId: string) {
    this.ws = ws;
    this.callSid = callSid;
    this.streamSid = streamSid;
    this.agentId = agentId;
    this.metrics = new MetricsService()
    this.retrievalService = new RetrievalService();
    // Initialize Static Services
    this.llmService = new LLMService();
    this.ttsService = new TtsService();
    
    // Initialize Session
    SessionService.initSession(this.callSid, this.agentId);
    
    this.setupPipeline();
  }

  private setupPipeline() {
    this.transcriptionService = new TranscriptionService(
      // 1. User Finished Speaking
      async (text) => this.handleUserMessage(text),
      // 2. User Interrupted
      () => this.handleInterruption()
    );
  }

  /**
   * Core Logic Loop: Ear -> Brain -> Mouth
   */
  private async handleUserMessage(userText: string) {
    if (!this.isAlive) return;

    this.currentInteractionId++;
    const myId = this.currentInteractionId;

    // ⏱️ START TIMER (User stopped speaking)
    this.metrics.startTurn(myId);

    try {

      // Search the database for relevant info before asking LLM
      const context = await this.retrievalService.search(userText);

      if (context) {
        logger.info(`🔍 Found ${context.length} relevant chunks in the database.`);
      } else {
        logger.info("🔍 No relevant chunks found in the database.");
      }

      // ⏱️ Mark LLM Start
      this.metrics.mark(myId, "llmStart");

      await this.llmService.generateResponse(userText, async (aiSentence) => {
        if (myId !== this.currentInteractionId) return;

        // ⏱️ Mark LLM First Token (We got the first sentence)
        this.metrics.mark(myId, "llmFirstToken");
        
        // ⏱️ Mark TTS Start
        this.metrics.mark(myId, "ttsStart");

        const audio = await this.ttsService.generateAudio(aiSentence);
        
        // ⏱️ Mark TTS First Byte (We got audio)
        this.metrics.mark(myId, "ttsFirstByte");

        if (myId !== this.currentInteractionId) return;

        if (audio) this.sendAudio(audio);
      });
    } catch (error) {
      logger.error({ error }, "Pipeline Error");
      this.playFallbackError();
    }
  }
  /**
   * Handle Barge-In
   */
  private handleInterruption() {
    logger.info(" Interruption detected");
    this.currentInteractionId++; // Invalidate pending actions
    this.sendClearMessage();
  }

  /**
   * Error Recovery: The "Safety Net"
   */
  private async playFallbackError() {
    logger.warn(" Triggering Fallback Audio");
    // In a real app, load a pre-recorded WAV file here.
    // For now, we try to generate a quick apology.
    try {
      const audio = await this.ttsService.generateAudio("I'm having trouble connecting. One moment.");
      if (audio) this.sendAudio(audio);
    } catch (e) {
      logger.error(" Critical: Even Fallback Failed");
    }
  }

  // --- WebSocket Helpers ---

  public handleAudio(payload: string) {
    if (this.transcriptionService) {
      this.transcriptionService.send(payload);
    }
  }

  private sendAudio(audio: Buffer) {
    const mediaMessage = {
      event: "media",
      streamSid: this.streamSid,
      media: { payload: audio.toString("base64") },
    };
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(mediaMessage));
    }
  }

  private sendClearMessage() {
    const clearMessage = { event: "clear", streamSid: this.streamSid };
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(clearMessage));
    }
  }

  public cleanup() {
    this.isAlive = false;
    if (this.transcriptionService) this.transcriptionService.close();
    logger.info(` Orchestrator cleaned up for ${this.callSid}`);
  }
}