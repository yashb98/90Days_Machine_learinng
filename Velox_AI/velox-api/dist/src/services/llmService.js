"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LLMService = void 0;
const logger_1 = require("../utils/logger");
const definitions_1 = require("../tools/definitions");
const registry_1 = require("../tools/registry");
class LLMService {
    constructor() {
        this.client = null;
        this.modelName = "gemini-2.5-flash";
        this.systemPrompt = `
      You are a helpful assistant named Velox.
      Tone: Professional but friendly.
      Constraint: Keep answers concise (under 2 sentences). 
      If you need to use a tool, do it silently.
    `;
    }
    // Helper to load the SDK dynamically (Lazy Loading)
    async getClient() {
        if (this.client)
            return this.client;
        // DYNAMIC IMPORT: This fixes the "require" error
        const genai = await import("@google/genai");
        this.client = new genai.GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });
        console.log("--------------------------------------------------");
        console.log("🛠️  LLM Service Initialized (Dynamic Import)");
        console.log(`🤖  Model: ${this.modelName}`);
        console.log("--------------------------------------------------");
        return this.client;
    }
    async generateResponse(input, onSentence, context = "") {
        try {
            // Ensure Client is Loaded before using it
            const client = await this.getClient();
            let instructions = this.systemPrompt;
            if (context) {
                instructions += `\n\n=== KNOWLEDGE BASE ===\n${context}\n======================`;
            }
            // Start Chat using new SDK pattern
            const chat = client.chats.create({
                model: this.modelName,
                config: {
                    systemInstruction: instructions,
                    tools: [{ functionDeclarations: definitions_1.tools }],
                },
            });
            // Use sendMessage instead of send
            let response = await chat.sendMessage({
                parts: [{ text: input }],
            });
            // Tool Loop
            let functionCalls = response.functionCalls;
            while (functionCalls && functionCalls.length > 0) {
                const call = functionCalls[0];
                const { name, args } = call;
                logger_1.logger.info(`AI wants to execute: ${name}(${JSON.stringify(args)})`);
                // @ts-ignore
                const functionToCall = registry_1.toolRegistry[name];
                if (functionToCall) {
                    const apiResult = await functionToCall(args);
                    logger_1.logger.info(`Tool Result: ${JSON.stringify(apiResult)}`);
                    response = await chat.sendMessage({
                        parts: [{
                                functionResponse: {
                                    name: name,
                                    response: apiResult,
                                },
                            }],
                    });
                    functionCalls = response.functionCalls;
                }
                else {
                    break;
                }
            }
            const text = response.text;
            if (text) {
                this.processBuffer(text, onSentence);
            }
        }
        catch (error) {
            logger_1.logger.error({ error }, "Error generating LLM response");
            // Detailed error logging for debugging
            console.error("❌ GEMINI ERROR (Raw):", error);
            console.error("❌ GEMINI ERROR (Dir):", JSON.stringify(error, Object.getOwnPropertyNames(error), 2));
            // Check for common issues
            if (!process.env.GEMINI_API_KEY) {
                console.error("❌ CRITICAL: GEMINI_API_KEY is missing from environment variables!");
            }
            onSentence("I'm having trouble connecting right now.");
        }
    }
    processBuffer(text, onSentence) {
        const sentences = text.match(/[^.?!]+[.?!]+|[^.?!]+$/g) || [text];
        sentences.forEach((sentence) => {
            const trimmed = sentence.trim();
            if (trimmed) {
                logger_1.logger.info(`🤖 AI (Speaking): ${trimmed}`);
                onSentence(trimmed);
            }
        });
    }
}
exports.LLMService = LLMService;
