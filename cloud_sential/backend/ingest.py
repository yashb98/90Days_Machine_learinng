import os
from dotenv import load_dotenv
from backend.rag.services import PyPDFLoaderService, GeminiEmbedderService, PineconeService

# Load Env Vars
load_dotenv()


def main():
    print("🚀 Starting Ingestion Pipeline...")

    # 1. Initialize Services
    loader = PyPDFLoaderService()
    embedder = GeminiEmbedderService()
    # Use a unique index name
    vector_db = PineconeService(index_name="cloud-sentinel-gemini")

    # 2. Load Data
    pdf_path = "backend/data/acme_security_standards_v2.pdf"
    if not os.path.exists(pdf_path):
        print(
            f"❌ Error: File not found at {pdf_path}. Run generate_pdf.py first.")
        return

    print(f"📄 Loading {pdf_path}...")
    documents = loader.load(pdf_path)
    print(f"   -> Split into {len(documents)} chunks.")

    # 3. Embed Data
    print("🧠 Generating Embeddings via Gemini...")
    try:
        texts = [doc.content for doc in documents]
        embeddings = embedder.embed_documents(texts)
    except Exception as e:
        print(f"❌ Error during embedding: {e}")
        return

    # 4. Store Data
    print("💾 Storing in Pinecone...")
    vector_db.upsert(documents, embeddings)

    print("✨ Pipeline Complete!")


if __name__ == "__main__":
    main()
