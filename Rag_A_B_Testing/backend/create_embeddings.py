import numpy as np
import pandas as pd
import os
import sys
import time
from dotenv import load_dotenv

# --- NEW: Import Ollama ---
import ollama
from ollama import Client

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EHR_CACHE_FILE = os.path.join(SCRIPT_DIR, 'ehr_cache.parquet')

# --- NEW: Configuration for local Ollama embedding model ---
OLLAMA_EMBED_MODEL_ID = "mxbai-embed-large"
# Ollama's Python library handles batching well.
BATCH_SIZE = 128


def init_ollama_client() -> Client | None:
    """Initializes and returns the Ollama Client."""
    try:
        # Connects to the default http://localhost:11434
        client = Client()
        # Test connection by listing local models
        client.list()
        print("Ollama Client initialized and connected.")
        return client
    except Exception as e:
        print("ERROR: Failed to connect to Ollama.", file=sys.stderr)
        print("Please ensure Ollama is running.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None


def embed_content_batch(client: Client, texts: list[str]) -> list[list[float]] | None:
    """
    Calls the local Ollama API to embed a batch of texts.

    Returns:
        A list of embeddings (list of floats), or None if an error occurred.
    """
    try:
        # The ollama.embed() function efficiently handles a list of inputs
        response = client.embed(
            model=OLLAMA_EMBED_MODEL_ID,
            input=texts
        )

        # The response is a dict with an 'embeddings' key
        return response['embeddings']

    except Exception as e:
        print(f"API Error embedding batch (Ollama): {e}", file=sys.stderr)
        return None


def main():
    """
    Main script to load data, generate embeddings via local Ollama, and save.
    """
    client = init_ollama_client()
    if not client:
        sys.exit(1)

    # 1. Load the existing cache file
    print(f"Loading data from {EHR_CACHE_FILE}...")
    if not os.path.exists(EHR_CACHE_FILE):
        print(
            f"ERROR: Cache file not found: {EHR_CACHE_FILE}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_parquet(EHR_CACHE_FILE)
    except Exception as e:
        print(f"Error reading Parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Check if embeddings already exist
    if 'embedding' in df.columns:
        print("'embedding' column already found. File is already processed.")
        print("To re-generate, please remove the 'embedding' column manually.")
        sys.exit(0)

    if 'content' not in df.columns:
        print("ERROR: 'content' column not found in cache file.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(df)} records without embeddings. Starting generation...")

    # 3. Get list of texts to embed
    texts_to_embed = df['content'].tolist()
    all_embeddings = []

    total_batches = -(-len(texts_to_embed) // BATCH_SIZE)  # Ceiling division
    start_time = time.time()

    # 4. Process in batches
    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch_texts = texts_to_embed[i: i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1

        print(f"Processing batch {current_batch_num}/{total_batches}...")

        batch_embeddings = embed_content_batch(client, batch_texts)

        if batch_embeddings is None:
            print(f"Failed to process batch {current_batch_num}. Aborting.")
            sys.exit(1)

        all_embeddings.extend(batch_embeddings)

    end_time = time.time()
    total_time = end_time - start_time
    records_per_sec = len(texts_to_embed) / total_time

    print("-" * 30)
    print(f"Embedding generation complete.")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Performance: {records_per_sec:.2f} records/second")
    print("-" * 30)

    # 5. Validate and save
    if len(all_embeddings) != len(df):
        print("ERROR: Mismatch in embedding count. Aborting save.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully generated {len(all_embeddings)} embeddings.")

    # Add the new column
    df['embedding'] = all_embeddings

    # Save the file back, overwriting the old one
    print(f"Saving updated DataFrame to {EHR_CACHE_FILE}...")
    try:
        df.to_parquet(EHR_CACHE_FILE, index=False)
        print("Success! Your cache file is now ready for the RAG application.")
    except Exception as e:
        print(f"Error saving updated Parquet file: {e}", file=sys.stderr)


if __name__ == "__main__":
    load_dotenv()
    main()
