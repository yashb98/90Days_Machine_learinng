import { logger } from "../utils/logger";
import { tools } from "../tools/definitions"; 
import { toolRegistry } from "../tools/registry";

export class LLMService {
  private client: any = null;
  // Use the reliable experimental model
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

  // Helper to load the SDK dynamically
  private async getClient(): Promise<any> {
    if (this.client) return this.client;
    
    // Dynamic import
    const { GoogleGenAI } = await import("@google/genai");
    
    this.client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    
    console.log("--------------------------------------------------");
    console.log("🛠️  LLM Service Initialized (New SDK)");
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
      const client = await this.getClient();

      let instructions = this.systemPrompt;
      if (context) {
        instructions += `\n\n=== KNOWLEDGE BASE ===\n${context}\n======================`;
      }

      // 1. Create Chat
    let response = await client.models.generateContent({
      model: this.modelName,
      contents: [
        {
          role: "user",
          parts: [{ text: input }],
        },
      ],
      config: {
        systemInstruction: instructions,
        tools: [{ functionDeclarations: tools }],
      },
    });


      // 3. Tool Loop
      let functionCalls = response.functionCalls;

      while (functionCalls && functionCalls.length > 0) {
        const call = functionCalls[0];
        const { name, args } = call;

        logger.info(`🤖 AI wants to execute: ${name}(${JSON.stringify(args)})`);

        // @ts-ignore
        const functionToCall = toolRegistry[name];

        if (functionToCall) {
          const apiResult = await functionToCall(args);
          logger.info(`Tool Result: ${JSON.stringify(apiResult)}`);

          // 4. Send Tool Result Back
          response = await client.models.generateContent({
            model: this.modelName,
            contents: [
              {
                role: "tool",
                parts: [
                  {
                    functionResponse: {
                      name,
                      response: apiResult,
                    },
                  },
                ],
              },
            ],
            config: {
              systemInstruction: instructions,
              tools: [{ functionDeclarations: tools }],
            },
          });
          
          functionCalls = response.functionCalls;
        } else {
          logger.warn(` Tool '${name}' not found.`);
          break;
        }
      }

      // 5. Final Output
      const text = response.text;
      if (text) {
        this.processBuffer(text, onSentence);
      }

    } catch (error: any) {
      logger.error({ error }, "Error generating LLM response");
      
      console.error("❌ GEMINI ERROR MESSAGE:", error.message);
      if (error.stack) console.error(error.stack);
      
      onSentence("I'm having trouble connecting right now.");
    }
  }

  private processBuffer(text: string, onSentence: (text: string) => void) {
    if (!text) return;
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