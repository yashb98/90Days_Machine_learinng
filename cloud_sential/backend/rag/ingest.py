import traceback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.rag.services import PyPDFLoaderService, GeminiEmbedderService, PineconeService

# This simulates a database of "Active Policies" for the frontend list
UPLOADED_FILES_DB = []


async def process_document(file_path: str, filename: str):
    """
    Orchestrates the full RAG pipeline: Load -> Split -> Embed -> Index
    Using Gemini Embeddings (768 Dimensions)
    """
    print(f"⚙️ PROCESSING: {filename}...")

    try:
        # 1. Load PDF
        # We assume the file is already saved to 'file_path' by main.py
        loader = PyPDFLoaderService()
        raw_docs = loader.load(file_path, split=False)

        # 2. Split (Sliding Window)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(raw_docs)
        print(f"   -> Split into {len(chunks)} chunks")

        # 3. Embed & Index
        print("   -> Generating Embeddings (Gemini)...")
        embedder = GeminiEmbedderService()

        # 👇 CHANGE: Instantiate without arguments.
        # This forces it to use the defaults (index='cloud-sentinel-gemini', dim=768)
        # defined in your services.py. This prevents mismatch errors.
        vector_db = PineconeService()

        # Batching logic (Critical for large PDFs)
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            # Prepare Text & Metadata
            texts = [doc.page_content for doc in batch]
            metadatas = [
                {
                    "source": filename,
                    "page": doc.metadata.get("page", 0),
                    "text": doc.page_content
                }
                for doc in batch
            ]

            # Generate Embeddings
            vectors = embedder.embed_documents(texts)

            # Zip into Pinecone format (ID, Vector, Metadata)
            to_upsert = [
                (f"{filename}_chunk_{i+j}", vec, meta)
                for j, (vec, meta) in enumerate(zip(vectors, metadatas))
            ]

            # Upsert Batch
            vector_db.index.upsert(vectors=to_upsert)
            print(f"   -> Batch {i//batch_size + 1} indexed.")

        # 4. Update "Database"
        policy_record = {
            "id": str(len(UPLOADED_FILES_DB) + 1),
            "name": filename,
            "status": "active",
            "lastUpdated": "Just now"
        }
        UPLOADED_FILES_DB.append(policy_record)

        print(f"✅ SUCCESS: {filename} is now in the Brain.")
        return policy_record

    except Exception as e:
        print(f"❌ ERROR processing {filename}:")
        # 👇 CHANGE: Print the full error trace so you can see WHY it failed
        traceback.print_exc()
        raise e
