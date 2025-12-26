import os
from dotenv import load_dotenv
from backend.rag.services import PineconeService, GeminiEmbedderService

load_dotenv()


def diagnose_brain():
    print("🔍 --- RAG DIAGNOSIS START ---")

    # 1. Check Pinecone Stats
    try:
        pc = PineconeService(index_name="cloud-sentinel-gemini")
        stats = pc.index.describe_index_stats()
        print(f"📊 Index Stats: {stats}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 2. Test a Query
    print("\n🧠 Testing Retrieval...")
    try:
        embedder = GeminiEmbedderService()
        query = "production bucket versioning"
        vector = embedder.embed_query(query)

        # This returns a list of 'Document' objects
        results = pc.search(vector, top_k=3)

        if not results:
            print("❌ Search returned ZERO results.")
        else:
            print(f"✅ Found {len(results)} matches!")
            for i, doc in enumerate(results):
                # Document objects have 'page_content', not 'score'
                print(
                    f"   [{i+1}] Content Preview: {doc.page_content[:100]}...")
                print(
                    f"       Source: {doc.metadata.get('source', 'Unknown')}")

    except Exception as e:
        print(f"❌ Search Failed: {e}")


if __name__ == "__main__":
    diagnose_brain()
    print("🔍 --- RAG DIAGNOSIS END ---")
