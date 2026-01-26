// test-embedding-only.ts
import dotenv from "dotenv";
dotenv.config();
import { embeddingService } from "./src/services/embeddingService";

async function test() {
  try {
    console.log("🧪 Testing embedding generation...\n");

    const text = "Order 554 is in transit.";
    console.log(`📝 Text: "${text}"`);
    
    const embedding = await embeddingService.getEmbedding(text);
    
    if (embedding) {
      console.log(`✅ Success! Generated ${embedding.length} dimensions`);
      console.log(`First 5 values: [${embedding.slice(0, 5).join(", ")}]`);
    } else {
      console.log("❌ Failed to generate embedding");
    }

  } catch (error) {
    console.error("❌ Error:", error);
  }
}

test();