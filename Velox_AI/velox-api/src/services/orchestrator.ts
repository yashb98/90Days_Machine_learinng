import WebSocket from "ws";
import { logger } from "../app";
import { TranscriptionService } from "./transcriptionService";
import { LLMService } from "./llmService";
import { TtsService } from "./ttsService";
import { SessionService } from "./sessionService";

export class CallOrchestrator {
  private ws: WebSocket;
  private callSid: string;
  private streamSid: string;
  private agentId: string;
  
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

    try {
      // 🧠 Brain
      await this.llmService.generateResponse(userText, async (aiSentence) => {
        if (myId !== this.currentInteractionId) return; // Interrupted?

        // 🗣️ Mouth
        const audio = await this.ttsService.generateAudio(aiSentence);
        
        if (myId !== this.currentInteractionId) return; // Interrupted?

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