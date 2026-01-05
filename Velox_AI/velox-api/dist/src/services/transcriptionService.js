"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TranscriptionService = void 0;
const sdk_1 = require("@deepgram/sdk");
const app_1 = require("../app");
class TranscriptionService {
    constructor(onTranscript) {
        this.onTranscript = onTranscript;
        const deepgram = (0, sdk_1.createClient)(process.env.DEEPGRAM_API_KEY || "");
        // Configure for Speed (Nova-2) and Phone Audio (Mulaw 8000Hz)
        this.deepgramLive = deepgram.listen.live({
            model: "nova-2",
            language: "en",
            encoding: "mulaw", // Twilio's format
            sample_rate: 8000, // Phone quality
            endpointing: 300, // Wait 300ms of silence to trigger "Final"
            smart_format: true,
            interim_results: true, // We want to see words AS they are spoken
        });
        this.setupEventListeners();
    }
    setupEventListeners() {
        this.deepgramLive.on(sdk_1.LiveTranscriptionEvents.Open, () => {
            app_1.logger.info("Deepgram Connection Opened");
        });
        this.deepgramLive.on(sdk_1.LiveTranscriptionEvents.Transcript, (data) => {
            const transcript = data.channel.alternatives[0].transcript;
            // 'is_final' means the user paused long enough (300ms)
            if (transcript && data.is_final) {
                app_1.logger.info(`USER (Final): ${transcript}`);
                this.onTranscript(transcript);
                // TODO: Send this text to the LLM (Day 7)
            }
            else if (transcript) {
                // This is interim results (flashy text)
                // logger.debug(`... ${transcript}`); 
            }
        });
        this.deepgramLive.on(sdk_1.LiveTranscriptionEvents.Error, (err) => {
            app_1.logger.error({ err }, "Deepgram Error");
        });
        // "UtteranceEnd" is the hard VAD signal - Silence detected
        this.deepgramLive.on(sdk_1.LiveTranscriptionEvents.UtteranceEnd, () => {
            app_1.logger.info("Silence Detected (Turn Finished)");
        });
    }
    /**
     * Send raw audio chunks from Twilio to Deepgram
     */
    send(payload) {
        // Twilio sends base64, Deepgram expects a Buffer
        const audioBuffer = Buffer.from(payload, "base64");
        if (this.deepgramLive.getReadyState() === 1) { // 1 = OPEN
            this.deepgramLive.send(audioBuffer);
        }
    }
    close() {
        app_1.logger.info("Closing Deepgram Connection");
        this.deepgramLive.finish();
    }
}
exports.TranscriptionService = TranscriptionService;
