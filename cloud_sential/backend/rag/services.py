import os
import time
import google.genai as genai
from typing import List
from .interfaces import IDocumentLoader, IEmbedder, IVectorStore, Document

# Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# --- 1. PDF Loader ---


class PyPDFLoaderService:
    def load(self, source: str, split: bool = True) -> List[Document]:
        loader = PyPDFLoader(source)
        raw_docs = loader.load()

        if not split:
            return raw_docs

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(raw_docs)

        return [Document(page_content=c.page_content, metadata=c.metadata) for c in chunks]

# --- 2. Gemini Embedder (The "Brain") ---


class GeminiEmbedderService(IEmbedder):
    def __init__(self):
        # Configure API
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is missing from .env")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = "models/text-embedding-004"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        # Batching to be safe with rate limits
        for text in texts:
            clean_text = text.replace("\n", " ")
            try:
                response = genai.embed_content(
                    model=self.model,
                    content=clean_text,
                    task_type="retrieval_document"
                )
                results.append(response['embedding'])
                time.sleep(0.05)  # Tiny pause for stability
            except Exception as e:
                print(f"⚠️ Embedding Error: {e}")
                # Return zero vector fallback to prevent crash (768 dims)
                results.append([0.0] * 768)

        return results

    def embed_query(self, text: str) -> List[float]:
        clean_text = text.replace("\n", " ")
        response = genai.embed_content(
            model=self.model,
            content=clean_text,
            task_type="retrieval_query"
        )
        return response['embedding']

# --- 3. Pinecone Store (The "Memory") ---


class PineconeService(IVectorStore):
    def __init__(self, index_name="cloud-sentinel-gemini", dimension=768):
        # ⚠️ GEMINI USES 768 DIMENSIONS. DO NOT CHANGE THIS.

        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY is missing from .env")

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Auto-Create Index if missing
        existing_indexes = self.pc.list_indexes().names()
        if index_name not in existing_indexes:
            print(
                f"⚙️ Creating new Pinecone index: {index_name} (Dim: {dimension})...")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(15)  # Wait for initialization

        self.index = self.pc.Index(index_name)

    def upsert(self, documents: List[Document], embeddings: List[List[float]]):
        vectors = []
        for i, (doc, vector) in enumerate(zip(documents, embeddings)):
            vector_id = f"chunk_{i}_{int(time.time())}"
            vectors.append({
                "id": vector_id,
                "values": vector,
                "metadata": {"text": doc.content, **doc.metadata}
            })

        # Batch upsert
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i+batch_size])
            print(f"   -> Upserted batch {i} to {i+batch_size}")

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Document]:
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return [
            Document(match['metadata']['text'], match['metadata'])
            for match in results['matches']
        ]
