import { createClient } from "@deepgram/sdk";
// import { ReadableStream } from "stream/web"; // Node 18+ specific
import { logger } from "../utils/logger";

export class TtsService {
  private deepgram: any;

  constructor() {
    this.deepgram = createClient(process.env.DEEPGRAM_API_KEY || "");
  }

  /**
   * Converts text to audio (Mulaw 8000Hz)
   */
  async generateAudio(text: string): Promise<Buffer | null> {
    try {
      const response = await this.deepgram.speak.request(
        { text },
        {
          model: "aura-asteria-en", // A fast, female voice (Try 'aura-helios-en' for male)
          encoding: "mulaw",        // Twilio requires mulaw
          sample_rate: 8000,        // Twilio requires 8000Hz
          container: "none",        // We need raw audio bytes, not a WAV file
        }
      );

      const stream = await response.getStream();
      if (stream) {
        const buffer = await this.streamToBuffer(stream);
        return buffer;
      }
      return null;
    } catch (error) {
      logger.error({ error }, "TTS Generation Error");
      return null;
    }
  }

  // Helper to convert readable stream to Buffer
  private async streamToBuffer(stream: ReadableStream): Promise<Buffer> {
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    return Buffer.concat(chunks);
  }
}