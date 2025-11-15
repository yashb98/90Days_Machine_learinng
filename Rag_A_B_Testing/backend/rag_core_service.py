import numpy as np
import pandas as pd
import requests
import os
import time
import sys
from typing import Dict, Any
from dotenv import load_dotenv

# --- NEW: Import Faiss ---
import faiss

# Libraries required for integration
from google import genai
from google.genai.errors import APIError
from google.genai import types

# --- Configuration and Environment Setup ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EHR_CACHE_FILE = os.path.join(SCRIPT_DIR, 'ehr_cache.parquet')

GEMINI_MODEL_ID = "models/gemini-pro-latest"
EMBEDDING_MODEL_ID = "models/embedding-001"

# --- Model Endpoints ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_ID = "mistral"

# Global cache for components
CONTEXT_DATA = None
VECTOR_STORE_INDEX = None  # This will become our Faiss index object
GEMINI_CLIENT = None


# ====================================================================
# ARCHITECTURE STEP 1: EMBEDDING & RETRIEVAL (Now with Faiss)
# ====================================================================

def init_retrieval_components():
    """
    Initializes the Gemini Client, loads the optimized Parquet cache,
    and builds the Faiss index from pre-computed embeddings.
    """
    global CONTEXT_DATA, VECTOR_STORE_INDEX, GEMINI_CLIENT

    # 1. Load .env file
    load_dotenv()

    # 2. Check for API Key
    if not os.environ.get('GEMINI_API_KEY'):
        print(
            "RAG Core Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        # ... (rest of error message)

    # 3. Instantiate the client
    try:
        GEMINI_CLIENT = genai.Client()
        print("RAG Core: Gemini Client initialized.")
    except Exception as e:
        print(
            f"RAG Core Error: Failed to initialize Gemini Client. Error: {e}", file=sys.stderr)
        return

    # 4. Load from the single optimized cache file
    print(f"RAG Core: Attempting to load optimized cache: {EHR_CACHE_FILE}")
    try:
        if not os.path.exists(EHR_CACHE_FILE):
            print(
                f"RAG Core Error: Cache file not found: {EHR_CACHE_FILE}", file=sys.stderr)
            CONTEXT_DATA = pd.DataFrame()
            return

        CONTEXT_DATA = pd.read_parquet(EHR_CACHE_FILE)
        print(
            f"RAG Core: Successfully loaded {len(CONTEXT_DATA)} patient records from cache.")

        # --- NEW: Critical Check for Embeddings ---
        # We MUST have a pre-computed 'embedding' column in the Parquet file.
        if 'embedding' not in CONTEXT_DATA.columns:
            print(
                "RAG Core Error: 'embedding' column not found in cache file.", file=sys.stderr)
            print(
                "Please run the pre-processing script to generate document embeddings.", file=sys.stderr)
            CONTEXT_DATA = pd.DataFrame()
            return

    except Exception as e:
        print(
            f"RAG Core Error: Could not read cache file {EHR_CACHE_FILE}. Error: {e}", file=sys.stderr)
        CONTEXT_DATA = pd.DataFrame()
        return

    # --- NEW: Build the Faiss Index ---
    try:
        print("RAG Core: Building Faiss index from cached embeddings...")

        # 1. Convert the 'embedding' column (list of floats) into a 2D numpy array
        # Faiss requires a float32 matrix
        embeddings = np.array(
            CONTEXT_DATA['embedding'].tolist()).astype('float32')

        # 2. Get the dimensionality of the vectors
        if len(embeddings.shape) != 2:
            print("RAG Core Error: Embedding array is not 2D.", file=sys.stderr)
            return
        d = embeddings.shape[1]  # e.g., 768 for embedding-001

        # 3. Create a simple Faiss index (IndexFlatL2 = exact, L2 distance)
        index = faiss.IndexFlatL2(d)

        # 4. Add the document embeddings to the index
        index.add(embeddings)

        # 5. Store the ready-to-use index globally
        VECTOR_STORE_INDEX = index
        print(
            f"Retrieval Component Ready: Faiss index built with {index.ntotal} vectors.")

    except Exception as e:
        print(
            f"RAG Core Error: Failed to build Faiss index. Error: {e}", file=sys.stderr)


# --- UPDATED: Renamed from 'simulation' to 'with_faiss' ---
def retrieve_context_with_faiss(query: str, top_k: int = 3) -> str:
    """
    Executes a live vector search using Gemini Embeddings and the Faiss index.
    """
    if GEMINI_CLIENT is None:
        return "Retrieval Failed: Gemini Client not initialized."
    # --- UPDATED: Check if the index is a valid Faiss object ---
    if not isinstance(VECTOR_STORE_INDEX, faiss.Index):
        return "Retrieval Failed: Faiss index is not built or data not loaded."

    try:
        # 1. Generate Query Embedding (Live API Call)
        embedding_result = GEMINI_CLIENT.models.embed_content(
            model=EMBEDDING_MODEL_ID,
            contents=query
        )

        # --- NEW: Faiss Search Logic ---
        # 2. Prepare the query vector for Faiss
        # Faiss search expects a 2D array: (number_of_queries, dimension)
        query_vector = np.array([embedding_result.embedding]).astype('float32')

        # 3. Perform the search
        # D = distances, I = indices (the 0-based row number in the original data)
        distances, indices = VECTOR_STORE_INDEX.search(query_vector, k=top_k)

        # 4. Retrieve the matching documents from the DataFrame
        # Get the list of indices for our single query
        retrieved_indices = indices[0]
        matches = CONTEXT_DATA.iloc[retrieved_indices]

        if matches.empty:
            return "No relevant context found in vector store."

        # --- END OF NEW LOGIC ---

        retrieved_chunks = [
            f"[Source {row['patient_id']}]: {row['content']}" for _, row in matches.iterrows()]

        return "\n\n---\n\n".join(retrieved_chunks)

    except APIError as e:
        return f"Retrieval Failed (Embedding API Error): {e}"
    except Exception as e:
        return f"Retrieval Failed (Local Error): {e}"


# ====================================================================
# ARCHITECTURE STEP 2: MODEL INFERENCE (Mistral-7B vs. Gemini Pro)
# ====================================================================

def call_mistral_7b(context: str, query: str) -> str:
    # ... (This function remains unchanged)
    prompt = f"Use the following EHR context ONLY to answer the question concisely and factually.\nCONTEXT:\n---\n{context}\n---\n\nQUESTION: {query}"
    payload = {
        "model": OLLAMA_MODEL_ID,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9}
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=20)
        response.raise_for_status()
        generated_text = response.json().get(
            "response", "Mistral output failed: No response field found.")
        return generated_text
    except requests.exceptions.ConnectionError:
        return "ERROR: Mistral-7B (Mode A) connection failed. Ensure Ollama is running on http://localhost:11434."
    except Exception as e:
        return f"ERROR: Mistral-7B API call failed. Details: {e}"


def call_gemini_pro(context: str, query: str) -> str:
    # ... (This function remains unchanged)
    if GEMINI_CLIENT is None:
        return "ERROR: Gemini Client not initialized. Check API Key environment variable."

    system_instruction = (
        "You are a highly capable medical assistant. Use the EHR context to answer. "
        "For complex assessment queries (like risk factors), synthesize the raw findings into structured HTML lists (<ul> tags). Your response must be grounded in the provided CONTEXT."
    )
    user_query = f"CONTEXT:\n---\n{context}\n---\n\nQUESTION: {query}"

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )
        return response.text
    except APIError as e:
        print(f"GemDini API Error (Generation): {e}", file=sys.stderr)
        return f"ERROR: Failed to connect to {GEMINI_MODEL_ID}. Check network or API Key. Error: {e}"
    except Exception as e:
        return f"ERROR: Unknown error during Gemini generation: {e}"


# ====================================================================
# MAIN EXPORTED FUNCTION (Used by Flask app.py)
# ====================================================================

def get_rag_response(query: str, mode: str) -> Dict[str, Any]:
    """Orchestrates the A/B test RAG pipeline."""
    start_time = time.time()

    if GEMINI_CLIENT is None or CONTEXT_DATA is None or CONTEXT_DATA.empty:
        print("RAG Core: Components not loaded. Attempting re-initialization.")
        init_retrieval_components()

        # ... (Error handling remains the same)
        if GEMINI_CLIENT is None:
            return {
                'mode': mode,
                'answer': "RAG System Failure: Gemini Client could not be initialized.",
                'context': "Error: See backend console. Check GEMINI_API_KEY.",
                'model_name': "System Failure",
                'latency_ms': f"{(time.time() - start_time) * 1000:.0f} ms"
            }
        if CONTEXT_DATA is None or CONTEXT_DATA.empty:
            return {
                'mode': mode,
                'answer': "RAG System Failure: Context data could not be loaded.",
                'context': "Error: See backend console. Check cache file (it may be missing the 'embedding' column).",
                'model_name': "System Failure",
                'latency_ms': f"{(time.time() - start_time) * 1000:.0f} ms"
            }

    # 1. Retrieval Layer (--- UPDATED ---)
    # Call the new Faiss-powered retrieval function
    context = retrieve_context_with_faiss(query, top_k=3)

    if "Retrieval Failed" in context or "ERROR" in context or "Failure" in context:
        return {
            'mode': mode,
            'answer': "RAG System Failure during context retrieval.",
            'context': context,  # This will contain the specific error message
            'model_name': "System Failure",
            'latency_ms': f"{(time.time() - start_time) * 1000:.0f} ms"
        }

    # 2. Model Selection & Generation (A/B Test)
    if mode == 'A':
        answer = call_mistral_7b(context, query)
        model_name = f"{OLLAMA_MODEL_ID} (Baseline/Stable - Local Ollama)"
    else:  # mode == 'B'
        answer = call_gemini_pro(context, query)
        model_name = f"{GEMINI_MODEL_ID} (Candidate/Reasoning - Live API Call)"

    end_time = time.time()

    # 3. Structure Output for API Gateway
    return {
        'mode': mode,
        'answer': answer,
        'context': context,
        'model_name': model_name,
        'latency_ms': f"{(end_time - start_time) * 1000:.0f} ms"
    }


# Ensure components are initialized when the service starts
try:
    init_retrieval_components()
except Exception as e:
    print(f"RAG Core: CRITICAL STARTUP ERROR. {e}", file=sys.stderr)
    pass
