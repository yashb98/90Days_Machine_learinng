import "dotenv/config";
import { LLMService } from "./src/services/llmService";

const brain = new LLMService();

console.log("User: Hello, who are you?");
brain.generateResponse("Hello, who are you?", (sentence) => {
    console.log(`>> AI: ${sentence}`);
});