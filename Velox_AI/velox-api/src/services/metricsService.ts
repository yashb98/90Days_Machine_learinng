import { logger } from "../utils/logger";

interface InteractionMetrics {
  id: number;
  startTime: number;      // When user finished speaking (STT Final)
  llmStartTime?: number;  // When we sent text to Gemini
  llmFirstToken?: number; // When Gemini sent the first chunk
  ttsStartTime?: number;  // When we sent text to Deepgram Aura
  ttsFirstByte?: number;  // When Audio came back
}

export class MetricsService {
  private metrics: Map<number, InteractionMetrics> = new Map();

  startTurn(interactionId: number) {
    this.metrics.set(interactionId, {
      id: interactionId,
      startTime: Date.now()
    });
  }

  mark(interactionId: number, stage: 'llmStart' | 'llmFirstToken' | 'ttsStart' | 'ttsFirstByte') {
    const m = this.metrics.get(interactionId);
    if (!m) return;

    if (stage === 'llmStart') m.llmStartTime = Date.now();
    if (stage === 'llmFirstToken' && !m.llmFirstToken) m.llmFirstToken = Date.now();
    if (stage === 'ttsStart') m.ttsStartTime = Date.now();
    if (stage === 'ttsFirstByte' && !m.ttsFirstByte) {
        m.ttsFirstByte = Date.now();
        this.logReport(interactionId); // Log as soon as we have audio
    }
  }

  private logReport(interactionId: number) {
    const m = this.metrics.get(interactionId);
    if (!m || !m.ttsFirstByte || !m.llmStartTime || !m.llmFirstToken) return;

    const report = {
      event: "latency_report",
      interactionId: m.id,
      // How long Gemini took to "think"
      llm_latency_ms: m.llmFirstToken - m.llmStartTime,
      // How long TTS took to generate audio
      tts_latency_ms: m.ttsFirstByte - (m.ttsStartTime || 0),
      // TOTAL Time from Silence -> AI Sound
      total_e2e_latency_ms: m.ttsFirstByte - m.startTime
    };

    logger.info(report, "Performance Metrics");
    
    // Clean up memory
    this.metrics.delete(interactionId);
  }
}