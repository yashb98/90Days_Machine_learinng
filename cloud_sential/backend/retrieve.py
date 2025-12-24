import os
from dotenv import load_dotenv
from backend.rag.services import GeminiEmbedderService, PineconeService

load_dotenv()


def main():
    # 1. Initialize
    embedder = GeminiEmbedderService()
    vector_db = PineconeService(index_name="cloud-sentinel-gemini")

    # 2. Define Query
    query = "What are the rules for Production S3 buckets?"
    print(f"Query: {query}\n")

    # 3. Search
    try:
        query_vector = embedder.embed_query(query)
        results = vector_db.search(query_vector, top_k=2)

        # 4. Display
        print("🔍 Search Results:")
        if not results:
            print("   (No results found)")

        for i, doc in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(doc.content)
            print("------------------\n")

    except Exception as e:
        print(f"Error during retrieval: {e}")


if __name__ == "__main__":
    main()
