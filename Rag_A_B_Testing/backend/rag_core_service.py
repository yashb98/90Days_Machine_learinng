import numpy as np
import pandas as pd
import requests
import os
import time
import sys
from typing import Dict, Any
from dotenv import load_dotenv

import faiss

# Libraries for Google API (Model B)
from google import genai
from google.genai.errors import APIError
from google.genai import types

# --- UPDATED: Import Ollama and Mistral AI (New Structure) ---
import ollama
from ollama import Client
from mistralai import Mistral

from sentence_transformers import CrossEncoder

# --- Configuration and Environment Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EHR_CACHE_FILE = os.path.join(SCRIPT_DIR, 'ehr_cache.parquet')

# --- Generative Models ---
GEMINI_MODEL_ID = "models/gemini-pro-latest"
MISTRAL_MODEL_ID = "mistral-tiny"  # Using Mistral 7B via their API

# --- Embedding Model (Ollama) ---
OLLAMA_EMBED_MODEL_ID = "mxbai-embed-large"
EMBEDDING_DIMENSION = 1024

# Global cache for components
CONTEXT_DATA = None
VECTOR_STORE_INDEX = None
GEMINI_CLIENT = None      # For Model B (Gemini)
OLLAMA_CLIENT = None      # For Embeddings
MISTRAL_CLIENT = None     # For Model A (Mistral)
RERANK_MODEL = None

# ====================================================================
# ARCHITECTURE STEP 1: EMBEDDING & RETRIEVAL (Ollama/Faiss)
# ====================================================================


def init_retrieval_components():
    """
    Initializes all clients, loads cache, builds Faiss index,
    and loads the reranking model.
    """
    global CONTEXT_DATA, VECTOR_STORE_INDEX, GEMINI_CLIENT, OLLAMA_CLIENT, MISTRAL_CLIENT, RERANK_MODEL

    # 1. Load .env file
    load_dotenv()

    # 2. Instantiate Google Gemini Client (for Model B)
    if os.environ.get('GEMINI_API_KEY'):
        try:
            GEMINI_CLIENT = genai.Client()
            print("RAG Core: Gemini Client initialized (for Model B).")
        except Exception as e:
            print(
                f"RAG Core Warning: Failed to init Gemini Client. Error: {e}", file=sys.stderr)
    else:
        print("RAG Core Warning: GEMINI_API_KEY not set. Model B will be unavailable.", file=sys.stderr)

    # 3. Instantiate Ollama Client (for Embeddings)
    try:
        OLLAMA_CLIENT = Client()
        OLLAMA_CLIENT.list()  # Test connection
        print("RAG Core: Ollama Client initialized (for Embeddings).")
    except Exception as e:
        print("RAG Core ERROR: Failed to connect to Ollama.", file=sys.stderr)
        print("Ensure Ollama is running. Retrieval will fail.", file=sys.stderr)
        return

    # 4. Instantiate Mistral AI Client (for Model A)
    api_key = os.environ.get('MISTRAL_API_KEY')
    if api_key:
        try:
            MISTRAL_CLIENT = Mistral(api_key=api_key)
            print("RAG Core: Mistral AI Client initialized (for Model A).")
        except Exception as e:
            print(
                f"RAG Core Warning: Failed to init Mistral Client. Error: {e}", file=sys.stderr)
    else:
        print("RAG Core Warning: MISTRAL_API_KEY not set. Model A (Mistral AI) will be unavailable.", file=sys.stderr)

    # Load reranking Model
    try:
        # This is a lightweight but powerful reranker.
        # It's not a medical-specific one, but it's a great start.
        RERANK_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("RAG Core: Reranking model (Cross-Encoder) loaded.")
    except Exception as e:
        print(
            f"RAG Core Error: Could not load reranking model. {e}", file=sys.stderr)
        # We can let the app continue, but reranking will fail

    # 5. Load Parquet cache file
    print(f"RAG Core: Attempting to load optimized cache: {EHR_CACHE_FILE}")
    try:
        CONTEXT_DATA = pd.read_parquet(EHR_CACHE_FILE)
        print(
            f"RAG Core: Successfully loaded {len(CONTEXT_DATA)} patient records from cache.")
        if 'embedding' not in CONTEXT_DATA.columns:
            print(
                "RAG Core Error: 'embedding' column not found in cache file.", file=sys.stderr)
            return
    except Exception as e:
        print(
            f"RAG Core Error: Could not read cache file {EHR_CACHE_FILE}. Error: {e}", file=sys.stderr)
        return

    # 6. Build the Faiss Index
    try:
        print("RAG Core: Building Faiss index from cached embeddings...")
        embeddings = np.array(
            CONTEXT_DATA['embedding'].tolist()).astype('float32')
        d = embeddings.shape[1]
        if d != EMBEDDING_DIMENSION:
            print(f"RAG Core Error: Embeddings dimension mismatch!", file=sys.stderr)
            return
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        VECTOR_STORE_INDEX = index
        print(
            f"Retrieval Component Ready: Faiss index built with {index.ntotal} vectors.")
    except Exception as e:
        print(
            f"RAG Core Error: Failed to build Faiss index. Error: {e}", file=sys.stderr)


# ... (Imports and Init functions remain the same) ...

# --- ⭐️ UPDATED FUNCTION: Returns Dict instead of String ⭐️ ---
def retrieve_context_with_faiss(query: str, top_k: int = 3) -> Dict[str, str]:
    """
    Executes retrieval + reranking and returns BOTH the HTML display
    and the raw text for the safety guardrail.
    """
    RETRIEVE_K = 10
    FINAL_K = top_k

    # Error handling returns empty raw text
    error_response = {"context_html": "", "raw_context_text": ""}

    if OLLAMA_CLIENT is None:
        error_response["context_html"] = "Retrieval Failed: Ollama Client not initialized."
        return error_response
    if not isinstance(VECTOR_STORE_INDEX, faiss.Index):
        error_response["context_html"] = "Retrieval Failed: Faiss index is not built."
        return error_response
    if RERANK_MODEL is None:
        error_response["context_html"] = "Retrieval Failed: Reranker model not loaded."
        return error_response

    try:
        # 1. Retrieve (Standard Faiss Logic)
        response = OLLAMA_CLIENT.embed(
            model=OLLAMA_EMBED_MODEL_ID, input=query)
        query_embedding = response['embeddings'][0]
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = VECTOR_STORE_INDEX.search(
            query_vector, k=RETRIEVE_K)

        matches = CONTEXT_DATA.iloc[indices[0]]
        if matches.empty:
            error_response["context_html"] = "No relevant context found."
            return error_response

        # 2. Rerank (Standard Cross-Encoder Logic)
        pairs = [[query, row['content']] for _, row in matches.iterrows()]
        scores = RERANK_MODEL.predict(pairs)
        scored_matches = list(zip(scores, matches.iterrows()))
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        final_matches = [row for score,
                         (index, row) in scored_matches[:FINAL_K]]

        # 3. Format Output (THE CHANGE IS HERE)

        # A. Create HTML for Frontend
        retrieved_chunks = [
            f"<li><b>[Source {row['patient_id']}]:</b> {row['content']}</li>"
            for row in final_matches
        ]
        context_html = "<ol>\n" + "\n".join(retrieved_chunks) + "\n</ol>"

        # B. Aggregate Raw Text for Guardrail
        raw_context_text = " ".join([row['content'] for row in final_matches])

        return {
            "context_html": context_html,
            "raw_context_text": raw_context_text
        }

    except Exception as e:
        error_response["context_html"] = f"Retrieval Failed (Local Error): {e}"
        return error_response


# ====================================================================
# ARCHITECTURE STEP 2: MODEL INFERENCE (Mistral-7B vs. Gemini Pro)
# ====================================================================


def call_mistral_ai_api(context: str, query: str) -> str:
    """
    Calls the official Mistral AI API (Mode A: Fast/Cloud).
    """
    if MISTRAL_CLIENT is None:
        return "ERROR: Mistral AI Client (Mode A) not initialized. Check MISTRAL_API_KEY."

    # ---  NEW, SMARTER SYSTEM PROMPT  ---
    system_prompt = (
        "You are a helpful medical assistant. Use the following EHR context ONLY to answer the question concisely. "
        "If the user asks for a list of items (like conditions, risks, or findings), you MUST categorize the results and "
        "structure your answer using simple HTML lists (<ul> and <li> tags) with headings (<h3>). "
        "Do not just output one flat list. Group related items logically."
    )
    user_prompt = f"CONTEXT:\n---\n{context}\n---\n\nQUESTION: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        chat_completion = MISTRAL_CLIENT.chat.complete(
            model=MISTRAL_MODEL_ID,
            messages=messages,
            temperature=0.2,
            top_p=0.9,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"ERROR: Mistral AI API call failed. Details: {e}"
# --- END OF UPDATED FUNCTION ---


def call_gemini_pro(context: str, query: str) -> str:
    """
    Calls the live Gemini API for generation (Mode B: Reasoning/Synthesis).
    """
    if GEMINI_CLIENT is None:
        return "ERROR: Gemini Client not initialized. Check API Key environment variable."

    # ---  NEW, SMARTER SYSTEM PROMPT  ---
    system_instruction = (
        "You are a highly capable medical assistant. Use the provided EHR CONTEXT ONLY to answer the question. "
        "Your response must be grounded in the context. "
        "**Crucially, if the user's question requires a list of items (e.g., 'what are...', 'list the...'), "
        "you MUST synthesize and categorize the raw findings into logical groups.** "
        "**Format these groups as structured HTML lists (e.g., '<h3>Category</h3><ul><li>Item 1</li><li>Item 2</li></ul>').** "
        "Do not output a single, flat list."
    )

    # Corrected the \N unicode error
    user_query = f"CONTEXT:\n---\n{context}\n---\n\\NQUESTION: {query}"

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
    except Exception as e:
        return f"ERROR: Unknown error during Gemini generation: {e}"
# --- END OF UPDATED FUNCTION ---


# ====================================================================
# MAIN EXPORTED FUNCTION (Used by Flask app.py)
# ====================================================================

def get_rag_response(query: str, mode: str) -> Dict[str, Any]:
    """Orchestrates the A/B test RAG pipeline."""
    start_time = time.time()

    if OLLAMA_CLIENT is None or CONTEXT_DATA is None or CONTEXT_DATA.empty:
        print("RAG Core: Components not loaded. Attempting re-initialization.")
        init_retrieval_components()
        if OLLAMA_CLIENT is None:
            return {
                'mode': mode,
                'answer': "RAG System Failure: Ollama Client (for embeddings) could not be initialized.",
                'context': "Error: See backend console. Is Ollama running?",
                'model_name': "System Failure",
                'latency_ms': f"{(time.time() - start_time) * 1000:.0f} ms"
            }
        if CONTEXT_DATA is None or CONTEXT_DATA.empty:
            return {
                'mode': mode,
                'answer': "RAG System Failure: Context data could not be loaded.",
                'context': "Error: See backend console. Check cache file path or run 'create_embeddings.py'.",
                'model_name': "System Failure",
                'latency_ms': f"{(time.time() - start_time) * 1000:.0f} ms"
            }

    # 1. Retrieval Layer
    # --- CHANGE 1: Unpack the dictionary returned by the new retrieve function ---
    retrieval_result = retrieve_context_with_faiss(query, top_k=3)

    # For the User/Frontend
    context_html = retrieval_result["context_html"]
    # For the LLM/Guardrail
    raw_context = retrieval_result["raw_context_text"]

    # --- CHANGE 2: Check for errors in the HTML string ---
    if "Retrieval Failed" in context_html or "ERROR" in context_html or "Failure" in context_html:
        return {
            'mode': mode,
            'answer': "RAG System Failure during context retrieval.",
            'context': context_html,
            'model_name': "System Failure",
            'latency_ms': f"{(time.time() - start_time) * 1000:.0f} ms"
        }

    # 2. Model Selection & Generation (A/B Test)
    # --- CHANGE 3: Pass 'raw_context' to the models, NOT the HTML ---
    if mode == 'A':
        if MISTRAL_CLIENT is None:
            answer = "ERROR: Mode A is unavailable. Mistral AI Client not initialized. Check API Key."
            model_name = "System Failure"
        else:
            answer = call_mistral_ai_api(raw_context, query)
            model_name = f"{MISTRAL_MODEL_ID} (Baseline/Stable - Mistral AI API)"
    else:  # mode == 'B'
        if GEMINI_CLIENT is None:
            answer = "ERROR: Mode B is unavailable. Gemini Client not initialized. Check API Key."
            model_name = "System Failure"
        else:
            answer = call_gemini_pro(raw_context, query)
            model_name = f"{GEMINI_MODEL_ID} (Candidate/Reasoning - Live API Call)"

    end_time = time.time()

    # 3. Structure Output for API Gateway
    # --- CHANGE 4: Return 'raw_context' so app.py can use it for the Guardrail ---
    return {
        'mode': mode,
        'answer': answer,
        'context': context_html,    # Send HTML to frontend for display
        'raw_context': raw_context,  # Send Raw Text to app.py for Guardrail check
        'model_name': model_name,
        'latency_ms': f"{(end_time - start_time) * 1000:.0f} ms"
    }


# Ensure components are initialized when the service starts
try:
    init_retrieval_components()
except Exception as e:
    print(f"RAG Core: CRITICAL STARTUP ERROR. {e}", file=sys.stderr)
    pass
