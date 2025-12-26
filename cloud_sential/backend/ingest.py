import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.rag.services import PyPDFLoaderService, GeminiEmbedderService, PineconeService

load_dotenv()


def ingest_advanced():
    # 1. Load the Big PDF
    pdf_path = "/Users/yashbishnoi/Downloads/Dundee university/90Days_Machine_learinng/cloud_sential/backend/data/acme_global_security_policy_v3.pdf"

    print(f"Loading {pdf_path}...")

    # Initialize your service
    loader_service = PyPDFLoaderService()

    # CRITICAL: We pass split=False to get full pages.
    # This allows us to apply the "Sliding Window" logic below.
    raw_docs = loader_service.load(pdf_path, split=False)

    print(f"   -> Loaded {len(raw_docs)} raw pages.")

    # 2. Sliding Window Splitter (The Advanced Logic)
    print("✂️ Splitting with Sliding Window (Size=600, Overlap=200)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # Capture full rules/paragraphs
        chunk_overlap=200,     # Overlap ensures no context is lost at cut points
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(raw_docs)
    print(f"   -> Created {len(chunks)} granular vector chunks.")

    # 3. Embed & Upsert
    print("Embedding & Indexing...")
    embedder = GeminiEmbedderService()
    vector_db = PineconeService(index_name="cloud-sentinel-gemini")

    # Batch upsert
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [{"source": "acme_v3",
                      "page": doc.metadata.get("page", 0),
                      "text": doc.page_content

                      }
                     for doc in batch]

        vectors = embedder.embed_documents(texts)

        to_upsert = [
            (f"chunk_{i+j}", vec, meta)
            for j, (vec, meta) in enumerate(zip(vectors, metadatas))
        ]

        vector_db.index.upsert(vectors=to_upsert)
        print(f"   -> Upserted batch {i}-{i+len(batch)}")

    print("Ingestion Complete!")


if __name__ == "__main__":
    ingest_advanced()
