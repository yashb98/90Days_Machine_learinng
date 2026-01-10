"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MetricsService = void 0;
const logger_1 = require("../utils/logger");
class MetricsService {
    constructor() {
        this.metrics = new Map();
    }
    startTurn(interactionId) {
        this.metrics.set(interactionId, {
            id: interactionId,
            startTime: Date.now()
        });
    }
    mark(interactionId, stage) {
        const m = this.metrics.get(interactionId);
        if (!m)
            return;
        if (stage === 'llmStart')
            m.llmStartTime = Date.now();
        if (stage === 'llmFirstToken' && !m.llmFirstToken)
            m.llmFirstToken = Date.now();
        if (stage === 'ttsStart')
            m.ttsStartTime = Date.now();
        if (stage === 'ttsFirstByte' && !m.ttsFirstByte) {
            m.ttsFirstByte = Date.now();
            this.logReport(interactionId); // Log as soon as we have audio
        }
    }
    logReport(interactionId) {
        const m = this.metrics.get(interactionId);
        if (!m || !m.ttsFirstByte || !m.llmStartTime || !m.llmFirstToken)
            return;
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
        logger_1.logger.info(report, "Performance Metrics");
        // Clean up memory
        this.metrics.delete(interactionId);
    }
}
exports.MetricsService = MetricsService;
