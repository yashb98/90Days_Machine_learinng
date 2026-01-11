import { RetrievalService } from "./src/services/retrievalService";
import { LLMService } from "./src/services/llmService";
import dotenv from "dotenv";

dotenv.config();

async function test() {
  const retrieval = new RetrievalService();
  const llm = new LLMService();

  const question = "Where did Yash go to university?"; 
  console.log(`\n❓ Asking: "${question}"...`);

  // 1. Search
  console.log("🔍 Searching database...");
  const context = await retrieval.search(question);
  
  // --- DEBUGGING: SEE WHAT THE DB FOUND ---
  console.log("\n📄 --- RETRIEVED CONTEXT (What the AI sees) ---");
  console.log(context ? context : "⚠️ EMPTY CONTEXT"); 
  console.log("-----------------------------------------------\n");
  // ----------------------------------------

  if (context) {
    console.log("✅ Context Found!");
  } else {
    console.log("❌ No Context Found (Is the PDF uploaded?)");
  }

  // 2. Ask LLM
  console.log("🧠 Generating Answer...");
  
  let fullAnswer = "";

  await llm.generateResponse(
    question, 
    (sentence) => {
      process.stdout.write(sentence + " "); 
      fullAnswer += sentence + " ";         
    }, 
    context
  );

  console.log("\n\n✅ Final Answer Captured:", fullAnswer);
  
  process.exit(0);
}

test();