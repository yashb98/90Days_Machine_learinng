// test-manual-ingestion.ts

import dotenv from "dotenv";
dotenv.config();

import { PrismaClient } from "@prisma/client";
import { embeddingService } from "./src/services/embeddingService";
import { ragService } from "./src/services/ragService";

const prisma = new PrismaClient();

async function main() {
  try {
    console.log("🚀 Manual RAG Ingestion Test\n");

    // Step 1: Setup
    let org = await prisma.organization.findFirst();
    if (!org) {
      org = await prisma.organization.create({
        data: {
          name: "Test Organization",
          slug: "test-org",
          api_key_hash: "test-hash-12345",
        },
      });
    }
    console.log(`✅ Organization: ${org.name}\n`);

    // Step 2: Create KB
    await prisma.knowledgeBase.deleteMany({
      where: { name: "Test E-commerce KB" },
    });

    const kb = await prisma.knowledgeBase.create({
      data: {
        name: "Test E-commerce KB",
        description: "Test KB",
        org_id: org.id,
      },
    });
    console.log(`✅ Knowledge Base: ${kb.id}\n`);

    // Step 3: Ingest documents ONE AT A TIME with delays
    const documents = [
      { content: "Order 554 is in transit. Expected delivery January 30.", metadata: { order_id: "554" } },
      { content: "To return a damaged product, contact support within 30 days.", metadata: { topic: "returns" } },
      { content: "Gaming Mouse Pro is in stock for $79.99.", metadata: { product: "gaming_mouse" } },
    ];

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      console.log(`\n📄 Document ${i + 1}/${documents.length}: "${doc.content.substring(0, 40)}..."`);
      
      try {
        // Generate embedding
        console.log("  🧠 Generating embedding...");
        const embedding = await embeddingService.getEmbedding(doc.content);
        
        if (!embedding) {
          console.log("  ❌ Embedding failed, skipping");
          continue;
        }
        
        console.log(`  ✅ Embedding generated (${embedding.length} dims)`);
        
        // Insert directly into database
        const embeddingStr = `[${embedding.join(",")}]`;
        
        await prisma.$executeRaw`
          INSERT INTO "knowledge_chunks" (id, content, embedding, metadata, kb_id, created_at, updated_at)
          VALUES (
            gen_random_uuid(),
            ${doc.content},
            ${embeddingStr}::vector,
            ${JSON.stringify(doc.metadata)}::jsonb,
            ${kb.id},
            NOW(),
            NOW()
          )
        `;
        
        console.log("  ✅ Chunk inserted");
        
        // Add delay between embeddings to avoid rate limits and memory buildup
        if (i < documents.length - 1) {
          console.log("  ⏳ Waiting 2 seconds...");
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
        
      } catch (error: any) {
        console.log(`  ❌ Error: ${error.message}`);
      }
    }

    // Step 4: Verify chunks were created
    const chunkCount = await prisma.knowledgeChunk.count({
      where: { kb_id: kb.id },
    });
    console.log(`\n✅ Total chunks in KB: ${chunkCount}\n`);

    if (chunkCount === 0) {
      console.log("❌ No chunks created. Exiting.");
      return;
    }

    // Step 5: Test searches
    console.log("=".repeat(70));
    console.log("🔍 Testing Hybrid Search\n");

    // Test 1
    console.log("📌 Test 1: 'What is the status of order 554?'");
    const context1 = await ragService.retrieveContext("What is the status of order 554?", kb.id, 2);
    if (context1) {
      console.log("\n📋 Context:\n" + context1);
    } else {
      console.log("❌ No context found");
    }

    // Test 2
    console.log("\n" + "=".repeat(70));
    console.log("\n📌 Test 2: 'How do I return a damaged product?'");
    const context2 = await ragService.retrieveContext("How do I return a damaged product?", kb.id, 2);
    if (context2) {
      console.log("\n📋 Context:\n" + context2);
    } else {
      console.log("❌ No context found");
    }

    // Test 3
    console.log("\n" + "=".repeat(70));
    console.log("\n📌 Test 3: 'Do you have gaming mice?'");
    const context3 = await ragService.retrieveContext("Do you have gaming mice?", kb.id, 2);
    if (context3) {
      console.log("\n📋 Context:\n" + context3);
    } else {
      console.log("❌ No context found");
    }

    console.log("\n" + "=".repeat(70));
    console.log("\n✅ All tests completed!\n");

  } catch (error) {
    console.error("\n❌ Error:", error);
  } finally {
    await prisma.$disconnect();
  }
}

main();