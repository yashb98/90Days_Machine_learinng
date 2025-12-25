import os
import time
from typing import List
import requests
from .interfaces import IDocumentLoader, IEmbedder, IVectorStore, Document

# Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain_core.embeddings import Embeddings


class PyPDFLoaderService:
    # Add 'split' parameter, defaulting to True (Preserves your existing behavior)
    def load(self, source: str, split: bool = True) -> List[Document]:
        loader = PyPDFLoader(source)
        raw_docs = loader.load()

        if not split:
            # Return raw pages so the advanced script can handle the windowing
            return raw_docs

        # Your existing default logic (500 chars)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        chunks = splitter.split_documents(raw_docs)

        return [Document(page_content=c.page_content, metadata=c.metadata) for c in chunks]

# 2. Gemini Embedder


class GeminiEmbedderService(IEmbedder):
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = "models/text-embedding-004"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batching prevents hitting rate limits if you have many chunks
        results = []
        for text in texts:
            # Clean newlines to improve embedding quality
            clean_text = text.replace("\n", " ")
            response = genai.embed_content(
                model=self.model,
                content=clean_text,
                task_type="retrieval_document"
            )
            results.append(response['embedding'])
            time.sleep(0.1)  # Brief pause for rate limits
        return results

    def embed_query(self, text: str) -> List[float]:
        clean_text = text.replace("\n", " ")
        response = genai.embed_content(
            model=self.model,
            content=clean_text,
            task_type="retrieval_query"
        )
        return response['embedding']

# 3. Pinecone Store


class PineconeService(IVectorStore):
    def __init__(self, index_name="cloud-sentinel-mistral"):  # New Index Name
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)

        # Check if index exists, create if not
        # Dimension 768 is specific to Gemini text-embedding-004
        if index_name not in self.pc.list_indexes().names():
            print(f"⚙️ Creating new Pinecone index: {index_name}...")
            self.pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(10)  # Wait for index to initialize

        self.index = self.pc.Index(index_name)

    def upsert(self, documents: List[Document], embeddings: List[List[float]]):
        vectors = []
        for i, (doc, vector) in enumerate(zip(documents, embeddings)):
            vector_id = f"chunk_{i}"
            vectors.append({
                "id": vector_id,
                "values": vector,
                "metadata": {"text": doc.content, **doc.metadata}
            })

        self.index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} chunks to Pinecone.")

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


class E5MistralService(Embeddings):
    def __init__(self):
        # OPTION A: Hugging Face API (Recommended for Laptops)
        self.api_url = "https://api-inference.huggingface.co/models/intfloat/e5-mistral-7b-instruct"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

    def embed_query(self, text: str) -> list[float]:
        """
        Embeds a query. Crucial: E5-Mistral needs an INSTRUCTION for queries.
        """
        # We explicitly add the instruction prompt as per documentation
        prompt = f"Instruct: Retrieve relevant security policy sections for this query\nQuery: {text}"
        return self._call_api(prompt)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embeds documents. Per docs, NO instruction needed for documents.
        """
        return [self._call_api(doc) for doc in documents]

    def _call_api(self, text: str) -> list[float]:
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": text, "options": {"wait_for_model": True}}
            )
            return response.json()
        except Exception as e:
            print(f"HF API Error: {e}")
            return []
