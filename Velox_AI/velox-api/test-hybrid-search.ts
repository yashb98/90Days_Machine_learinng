// test-hybrid-search.ts

import { ragService } from "./src/services/ragService";

async function test() {
  const kbId = "test-kb-id"; // You'll need to create a KB first

  console.log("\n🧪 Test 1: Specific ID Query");
  const context1 = await ragService.retrieveContext(
    "What is the status of order 554?",
    kbId,
    3
  );
  console.log(context1);

  console.log("\n🧪 Test 2: Conceptual Query");
  const context2 = await ragService.retrieveContext(
    "How do I return a damaged product?",
    kbId,
    3
  );
  console.log(context2);
}

test();