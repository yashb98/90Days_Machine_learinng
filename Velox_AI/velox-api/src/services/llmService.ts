// 1. Use 'import type' so it doesn't crash on startup
import type { GoogleGenAI } from "@google/genai";
import { logger } from "../utils/logger";
import { tools } from "../tools/definitions"; 
import { toolRegistry } from "../tools/registry";

export class LLMService {
  private client: GoogleGenAI | null = null;
  // ✅ NOW we can use the latest model!
  private modelName: string = "gemini-2.0-flash-exp"; 
  private systemPrompt: string;

  constructor() {
    this.systemPrompt = `
      You are a helpful assistant named Velox.
      Tone: Professional but friendly.
      Constraint: Keep answers concise (under 2 sentences). 
      If you need to use a tool, do it silently.
    `;
  }

  // 2. Helper to load the SDK dynamically (Lazy Loading)
  private async getClient(): Promise<GoogleGenAI> {
    if (this.client) return this.client;
    
    // ⚠️ DYNAMIC IMPORT: This fixes the "require" error
    const { GoogleGenAI } = await import("@google/genai");
    
    this.client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    
    console.log("--------------------------------------------------");
    console.log("🛠️  LLM Service Initialized (Dynamic Import)");
    console.log(`🤖  Model: ${this.modelName}`);
    console.log("--------------------------------------------------");
    
    return this.client;
  }

  async generateResponse(
    input: string, 
    onSentence: (text: string) => void, 
    context: string = "" 
  ) {
    try {
      // 3. Ensure Client is Loaded before using it
      const client = await this.getClient();

      let instructions = this.systemPrompt;
      if (context) {
        instructions += `\n\n=== KNOWLEDGE BASE ===\n${context}\n======================`;
      }

      // 4. Start Chat
      const chat = client.chats.create({
        model: this.modelName,
        config: {
          systemInstruction: instructions,
          tools: [{ functionDeclarations: tools }], 
        },
      });

      let response = await chat.send({
        model: this.modelName,
        config: { outputModalities: ["TEXT"] },
        parts: [{ text: input }],
      });

      // 5. Tool Loop
      let functionCalls = response.functionCalls;

      while (functionCalls && functionCalls.length > 0) {
        const call = functionCalls[0];
        const { name, args } = call;

        logger.info(`🤖 AI wants to execute: ${name}(${JSON.stringify(args)})`);

        // @ts-ignore
        const functionToCall = toolRegistry[name];

        if (functionToCall) {
          const apiResult = await functionToCall(args);
          logger.info(`✅ Tool Result: ${JSON.stringify(apiResult)}`);

          response = await chat.send({
            parts: [{
                functionResponse: {
                  name: name,
                  response: apiResult,
                },
            }],
          });
          functionCalls = response.functionCalls;
        } else {
          break;
        }
      }

      const text = response.text;
      if (text) {
        this.processBuffer(text, onSentence);
      }

    } catch (error: any) {
      logger.error({ error }, "Error generating LLM response");
      console.error("❌ GEMINI ERROR:", JSON.stringify(error, null, 2));
      onSentence("I'm having trouble connecting right now.");
    }
  }

  private processBuffer(text: string, onSentence: (text: string) => void) {
    const sentences = text.match(/[^.?!]+[.?!]+|[^.?!]+$/g) || [text];
    sentences.forEach((sentence) => {
      const trimmed = sentence.trim();
      if (trimmed) {
        logger.info(`🤖 AI (Speaking): ${trimmed}`);
        onSentence(trimmed);
      }
    });
  }
}