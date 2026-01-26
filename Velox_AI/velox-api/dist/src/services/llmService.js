"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LLMService = void 0;
const logger_1 = require("../utils/logger");
const definitions_1 = require("../tools/definitions");
const registry_1 = require("../tools/registry");
const FILLER_PHRASES = [
    "One moment, let me check that for you.",
    "Just a second, looking that up.",
    "Let me see what I can find.",
    "Checking on that now.",
];
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
    async getClient() {
        if (this.client)
            return this.client;
        const { GoogleGenAI } = await import("@google/genai");
        this.client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
        console.log("--------------------------------------------------");
        console.log("🛠️  LLM Service Initialized (@google/genai SDK)");
        console.log(`🤖  Model: ${this.modelName}`);
        console.log("--------------------------------------------------");
        return this.client;
    }
    async generateResponse(input, onSentence, context = "") {
        try {
            const ai = await this.getClient();
            let instructions = this.systemPrompt;
            if (context) {
                instructions += `\n\n=== KNOWLEDGE BASE ===\n${context}\n======================`;
            }
            // Convert tools to correct format for @google/genai
            const formattedTools = definitions_1.tools.map(tool => ({
                type: 'function',
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters, // Use 'parameters' not 'parametersJsonSchema'
            }));
            //  Use models.generateContent (NOT interactions)
            let response = await ai.models.generateContent({
                model: this.modelName,
                contents: input,
                config: {
                    systemInstruction: instructions,
                    tools: formattedTools,
                },
            });
            //  Tool Execution Loop
            while (response.functionCalls && response.functionCalls.length > 0) {
                const call = response.functionCalls[0];
                const { name, args } = call;
                logger_1.logger.info(`🤖 AI wants to execute: ${name}(${JSON.stringify(args)})`);
                // @ts-ignore
                const functionToCall = registry_1.toolRegistry[name];
                if (functionToCall) {
                    // Play Filler Phrase
                    const randomFiller = FILLER_PHRASES[Math.floor(Math.random() * FILLER_PHRASES.length)];
                    logger_1.logger.info(`🤖 AI (Filler): ${randomFiller}`);
                    onSentence(randomFiller);
                    // Execute Tool
                    const apiResult = await functionToCall(args);
                    logger_1.logger.info(`Tool Result: ${JSON.stringify(apiResult)}`);
                    //  Send function response back
                    response = await ai.models.generateContent({
                        model: this.modelName,
                        contents: [
                            {
                                role: 'user',
                                parts: [{ text: input }]
                            },
                            {
                                role: 'model',
                                parts: response.functionCalls.map((fc) => ({
                                    functionCall: fc
                                }))
                            },
                            {
                                role: 'function',
                                parts: [{
                                        functionResponse: {
                                            name: name,
                                            response: apiResult,
                                        }
                                    }]
                            }
                        ],
                        config: {
                            systemInstruction: instructions,
                            tools: formattedTools,
                        },
                    });
                }
                else {
                    logger_1.logger.warn(`❌ Tool '${name}' not found.`);
                    break;
                }
            }
            //  Extract final text response
            const text = response.text;
            if (text) {
                this.processBuffer(text, onSentence);
            }
        }
        catch (error) {
            logger_1.logger.error({ error }, "Error generating LLM response");
            console.error(" GEMINI ERROR MESSAGE:", error.message);
            if (error.stack)
                console.error(error.stack);
            onSentence("I'm having trouble connecting right now.");
        }
    }
    processBuffer(text, onSentence) {
        if (!text)
            return;
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
