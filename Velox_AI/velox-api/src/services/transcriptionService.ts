import { createClient, LiveClient, LiveTranscriptionEvents } from "@deepgram/sdk";
import { logger } from "../utils/logger";

type TranscriptCallback = (text: string) => void;
type InterruptCallback = () => void;

export class TranscriptionService {
  private deepgramLive: LiveClient;
  private onTranscript: TranscriptCallback;
  private onInterrupt: InterruptCallback;

 constructor(onTranscript: TranscriptCallback, onInterrupt: InterruptCallback) {
    this.onTranscript = onTranscript;
    this.onInterrupt = onInterrupt;
    const deepgram = createClient(process.env.DEEPGRAM_API_KEY || "");
    
    // Configure for Speed (Nova-2) and Phone Audio (Mulaw 8000Hz)
    this.deepgramLive = deepgram.listen.live({
      model: "nova-2",
      language: "en",
      encoding: "mulaw", // Twilio's format
      sample_rate: 8000, // Phone quality
      endpointing: 300,  // Wait 300ms of silence to trigger "Final"
      smart_format: true,
      interim_results: true, // We want to see words AS they are spoken
      vad_events: true,      // Voice Activity Detection events
    });

    this.setupEventListeners();
  }

  private setupEventListeners() {
    this.deepgramLive.on(LiveTranscriptionEvents.Open, () => {
      logger.info("Deepgram Connection Opened");
    });

    this.deepgramLive.on(LiveTranscriptionEvents.Transcript, (data) => {
      const transcript = data.channel.alternatives[0].transcript;
      this.onInterrupt();
      
      // 'is_final' means the user paused long enough (300ms)
      if (transcript && data.is_final) {
        logger.info(`USER (Final): ${transcript}`);
        this.onTranscript(transcript);
        // TODO: Send this text to the LLM (Day 7)
      } else if (transcript) {
        // This is interim results (flashy text)
        // logger.debug(`... ${transcript}`); 
      }
    });

    this.deepgramLive.on(LiveTranscriptionEvents.Error, (err) => {
      logger.error({ err }, "Deepgram Error");
    });
    
    // "UtteranceEnd" is the hard VAD signal - Silence detected
    this.deepgramLive.on(LiveTranscriptionEvents.UtteranceEnd, () => {
       logger.info("Silence Detected (Turn Finished)");
    });
  }

  /**
   * Send raw audio chunks from Twilio to Deepgram
   */
  public send(payload: string) {
    // Twilio sends base64, Deepgram expects a Buffer
    const audioBuffer = Buffer.from(payload, "base64");
    
    if (this.deepgramLive.getReadyState() === 1) { // 1 = OPEN
      this.deepgramLive.send(audioBuffer as any);
    }
  }

  public close() {
    logger.info("Closing Deepgram Connection");
    this.deepgramLive.finish();
  }
}