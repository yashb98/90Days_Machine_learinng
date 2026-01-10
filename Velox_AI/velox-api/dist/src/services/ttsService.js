"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TtsService = void 0;
const sdk_1 = require("@deepgram/sdk");
// import { ReadableStream } from "stream/web"; // Node 18+ specific
const logger_1 = require("../utils/logger");
class TtsService {
    constructor() {
        this.deepgram = (0, sdk_1.createClient)(process.env.DEEPGRAM_API_KEY || "");
    }
    /**
     * Converts text to audio (Mulaw 8000Hz)
     */
    async generateAudio(text) {
        try {
            const response = await this.deepgram.speak.request({ text }, {
                model: "aura-asteria-en", // A fast, female voice (Try 'aura-helios-en' for male)
                encoding: "mulaw", // Twilio requires mulaw
                sample_rate: 8000, // Twilio requires 8000Hz
                container: "none", // We need raw audio bytes, not a WAV file
            });
            const stream = await response.getStream();
            if (stream) {
                const buffer = await this.streamToBuffer(stream);
                return buffer;
            }
            return null;
        }
        catch (error) {
            logger_1.logger.error({ error }, "TTS Generation Error");
            return null;
        }
    }
    // Helper to convert readable stream to Buffer
    async streamToBuffer(stream) {
        const reader = stream.getReader();
        const chunks = [];
        while (true) {
            const { done, value } = await reader.read();
            if (done)
                break;
            chunks.push(value);
        }
        return Buffer.concat(chunks);
    }
}
exports.TtsService = TtsService;
