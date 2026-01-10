import { RetrievalService } from "./src/services/retrievalService";
import { LLMService } from "./src/services/llmService";
import dotenv from "dotenv";

dotenv.config();

async function test() {
  const retrieval = new RetrievalService();
  const llm = new LLMService();

  const question = "Where did Yash go to university?"; // Ask something from your resume!
  console.log(`\n❓ Asking: "${question}"...`);

  // 1. Search
  console.log("🔍 Searching database...");
  const context = await retrieval.search(question);
  
  if (context) {
    console.log("✅ Context Found!");
  } else {
    console.log("❌ No Context Found (Is the PDF uploaded?)");
  }

  // 2. Ask LLM
  console.log("🧠 Generating Answer...");
// Create a variable to hold the full answer
  let fullAnswer = "";

  // Call the function with the correct arguments
  await llm.generateResponse(
    question, 
    (sentence) => {
      // This callback runs every time the AI finishes a sentence
      process.stdout.write(sentence + " "); // Print directly to console
      fullAnswer += sentence + " ";         // Accumulate full text
    }, 
    context
  );

  console.log("\n\n Final Answer Captured:", fullAnswer);
  
  
  process.exit(0);
}

test();