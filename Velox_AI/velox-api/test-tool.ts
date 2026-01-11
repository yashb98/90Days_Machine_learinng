import { LLMService } from "./src/services/llmService";
import dotenv from "dotenv";

dotenv.config();

async function test() {
  const llm = new LLMService();

  // Test Case 1: Checking Order Status
  const query = "Can you check the status of order 123?";
  
  console.log(`\n❓ User: "${query}"`);
  console.log("-----------------------------------------");

  await llm.generateResponse(query, (sentence) => {
    console.log(`\n🗣️ AI Said: "${sentence}"`);
  });
  
  // Test Case 2: Checking Stock (Something that doesn't exist)
  const query2 = "Do we have any gaming mice in stock?";
  console.log(`\n\n❓ User: "${query2}"`);
  console.log("-----------------------------------------");

  await llm.generateResponse(query2, (sentence) => {
    console.log(`\n🗣️ AI Said: "${sentence}"`);
  });
}

test();