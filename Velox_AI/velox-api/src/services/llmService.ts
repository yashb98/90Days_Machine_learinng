import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import { logger } from "../app";

export class LLMService {
  private genAI: GoogleGenerativeAI;
  private model: GenerativeModel;
  private systemPrompt: string;

  constructor() {
    this.genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");
    this.model = this.genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    
    // Default Persona (We will make this dynamic later)
    this.systemPrompt = `
      You are a helpful assistant named Velox.
      Tone: Professional but friendly.
      Constraint: Keep answers concise (under 2 sentences). 
      Do NOT use emojis. Spoken responses only.
    `;
  }

  /**
   * Generates a streaming response from Gemini
   * @param input The user's spoken text
   * @param onSentence A callback function that triggers whenever a full sentence is ready
   */
  async generateResponse(input: string, onSentence: (text: string) => void) {
    try {
      const result = await this.model.generateContentStream({
        contents: [
          { role: "user", parts: [{ text: this.systemPrompt + "\nUser: " + input }] }
        ],
      });

      let buffer = "";

      for await (const chunk of result.stream) {
        const text = chunk.text();
        buffer += text;

        // Check if we have a complete sentence/phrase
        // We look for punctuation: . ? ! 
        const punctuationRegex = /[.?!]+/;
        const match = buffer.match(punctuationRegex);

        if (match && match.index !== undefined) {
          const splitIndex = match.index + match[0].length;
          const sentence = buffer.slice(0, splitIndex).trim();
          
          // Send the complete sentence to be spoken
          if (sentence) {
            logger.info(`🤖 AI (Thinking): ${sentence}`);
            onSentence(sentence);
          }
          
          // Keep the remainder in the buffer
          buffer = buffer.slice(splitIndex);
        }
      }

      // Flush any remaining text in the buffer
      if (buffer.trim()) {
        logger.info(`🤖 AI (Final): ${buffer.trim()}`);
        onSentence(buffer.trim());
      }

    } catch (error) {
      logger.error({ error }, "Error generating LLM response");
    }
  }
}