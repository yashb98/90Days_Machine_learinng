"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LLMService = void 0;
const generative_ai_1 = require("@google/generative-ai");
const logger_1 = require("../utils/logger");
class LLMService {
    constructor() {
        this.genAI = new generative_ai_1.GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");
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
     * @param context (Optional) Retrieved knowledge base content from RAG
     */
    async generateResponse(input, onSentence, context = "") {
        try {
            // 1. Dynamically inject context if available
            let currentPrompt = this.systemPrompt;
            if (context) {
                currentPrompt += `
        \n\n=== RELEVANT KNOWLEDGE BASE ===
        ${context}
        ===============================
        Use the knowledge base above to answer the user's question. 
        If the answer is not in the context, say "I'm sorry, I don't have that information in my records."
        `;
            }
            // 2. Send the request
            const result = await this.model.generateContentStream({
                contents: [
                    { role: "user", parts: [{ text: currentPrompt + "\nUser: " + input }] }
                ],
            });
            let buffer = "";
            for await (const chunk of result.stream) {
                const text = chunk.text();
                buffer += text;
                // Check if we have a complete sentence/phrase
                const punctuationRegex = /[.?!]+/;
                const match = buffer.match(punctuationRegex);
                if (match && match.index !== undefined) {
                    const splitIndex = match.index + match[0].length;
                    const sentence = buffer.slice(0, splitIndex).trim();
                    if (sentence) {
                        logger_1.logger.info(`🤖 AI (Thinking): ${sentence}`);
                        onSentence(sentence);
                    }
                    buffer = buffer.slice(splitIndex);
                }
            }
            if (buffer.trim()) {
                logger_1.logger.info(`🤖 AI (Final): ${buffer.trim()}`);
                onSentence(buffer.trim());
            }
        }
        catch (error) {
            logger_1.logger.error({ error }, "Error generating LLM response");
        }
    }
}
exports.LLMService = LLMService;
