# rag.py
import os
import re
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader


# -------------------------------
#  CONFIGURATION
# -------------------------------

# Make sure to set GOOGLE_API_KEY in environment
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

CHROMA_DB_PATH = "rag_vectorstore"  # can be in-memory or disk-persisted
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)


# -------------------------------
#  PDF LOADING + CHUNKING
# -------------------------------

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF and split into chunks."""

    reader = PdfReader(file_path)
    all_chunks = []
    chunk_id = 1

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        splits = text_splitter.split_text(text)
        for chunk in splits:
            all_chunks.append({
                "page": i + 1,
                "chunk_id": chunk_id,
                "text": chunk
            })
            chunk_id += 1

    return all_chunks


# -------------------------------
#  VECTORSTORE SETUP
# -------------------------------
def create_vectorstore(chunks: List[Dict[str, Any]]):
    """Create or refresh a ChromaDB collection from chunks."""
    client = chromadb.Client()
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # Drop existing collection if it exists
    if "pdf_chunks" in [c.name for c in client.list_collections()]:
        client.delete_collection("pdf_chunks")

    # Create a new collection
    collection = client.create_collection(
        name="pdf_chunks",
        embedding_function=embedding_fn
    )

    # Add new chunks
    collection.add(
        ids=[str(c["chunk_id"]) for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"page": c["page"], "chunk_id": c["chunk_id"]}
                   for c in chunks]
    )

    print(f" Loaded {len(chunks)} chunks into Chroma vectorstore.")
    return collection


def retrieve_chunks(query: str, k: int = 3, collection=None) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks from Chroma vector DB."""
    if not collection:
        client = chromadb.Client()
        collection = client.get_collection("pdf_chunks")

    results = collection.query(query_texts=[query], n_results=k)
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "page": results["metadatas"][0][i]["page"],
            "chunk_id": results["metadatas"][0][i]["chunk_id"],
            "text": results["documents"][0][i],
            "score": results["distances"][0][i] if "distances" in results else 0
        })
    return retrieved


# -------------------------------
#  PROMPT BUILDING + GENERATION
# -------------------------------

def sanitize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    return text.strip()


def build_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join([
        f"[Page {r['page']}, Chunk {r['chunk_id']}]\n{sanitize_text(r['text'])}"
        for r in retrieved_chunks
    ])
    return f"""
You are a helpful assistant. Use the provided context to answer the user's question.
If not found, then look for it again."

Context:
{context}

Question: {query}

Answer:
"""


def generate_answer(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    model = genai.GenerativeModel(model_name)
    print(model.generate_content("Hello from Gemini!").text)
    response = model.generate_content([{"text": prompt}])
    return getattr(response, "text", str(response))


def rag_query(query: str, k: int = 3, collection=None) -> Dict[str, Any]:
    retrieved = retrieve_chunks(query, k, collection)
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt)
    return {"query": query, "answer": answer, "sources": retrieved}
