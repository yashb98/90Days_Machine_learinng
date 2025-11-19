import re  # Ensure 're' is imported at the top
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv  # 1. Utility to load environment variables from .env
import os
import sys  # Added for path correction
from typing import Dict, Any, Set
import spacy
import json
import time


# --- Load Environment Variables from .env File (MUST BE FIRST) ---
load_dotenv()

# --- Path Setup ---
# This ensures we can import from 'rag_core_service.py'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
# --- End Path Setup ---
DATASET_FILE = os.path.join(SCRIPT_DIR, 'fine_tuning_dataset.jsonl')
# 2. Now import the RAG Core Service.
try:
    from rag_core_service import get_rag_response
except ImportError as e:
    print("-------------------------------------------------------", file=sys.stderr)
    print(f"FATAL ERROR: Could not import from 'rag_core_service.py'.", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    print("Ensure 'app.py' and 'rag_core_service.py' are in the same directory.", file=sys.stderr)
    print("-------------------------------------------------------", file=sys.stderr)
    sys.exit(1)


# --- NEW: Evaluation Framework Setup (scispaCy) ---
try:
    # Load the small scispaCy model for entity extraction
    NLP_MODEL = spacy.load("en_core_sci_sm")
    print("Evaluation: scispaCy model 'en_core_sci_sm' loaded.")
except IOError:
    print("Evaluation Error: Could not load 'en_core_sci_sm'.", file=sys.stderr)
    print("Run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz", file=sys.stderr)
    NLP_MODEL = None

# This is your "Ground Truth" database.
# You must build this manually for your test queries.
GROUND_TRUTH_DB = {
    "what is the medication for hypertension?": [
        "lisinopril", "captopril", "amlodipine", "verapamil",
        "hydrochlorothiazide", "furosemide", "sacubitril", "valsartan",
        "losartan", "carvedilol"
    ],
    "what are risk factors for stroke?": [
        "hypertension", "atrial fibrillation", "obesity", "hyperlipidemia", "smoking"
    ]
    # Add other test queries and their "ideal" concepts here
}


def get_entities(text: str) -> Set[str]:
    """Uses scispaCy to extract all unique medical entities from text."""
    if NLP_MODEL is None:
        return set()

    doc = NLP_MODEL(text)
    # We lowercase to normalize the concepts
    return set([ent.text.lower() for ent in doc.ents])


def calculate_concept_f1(generated_text: str, ground_truth_key: str) -> Dict[str, Any]:
    """Calculates Precision, Recall, and F1 for clinical concepts."""

    # 1. Get ground truth (GT) entities
    # Normalize the query key to match the database
    ground_truth_concepts = set(
        GROUND_TRUTH_DB.get(ground_truth_key.lower(), []))
    if not ground_truth_concepts:
        return {"error": "No ground truth found for this query.", "f1_score": 0, "precision": 0, "recall": 0}

    # 2. Get generated entities
    generated_concepts = get_entities(generated_text)

    # 3. Calculate F1 metrics
    tp = len(generated_concepts.intersection(ground_truth_concepts))
    fp = len(generated_concepts.difference(ground_truth_concepts))
    fn = len(ground_truth_concepts.difference(generated_concepts))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "true_positives": list(generated_concepts.intersection(ground_truth_concepts)),
        "false_positives": list(generated_concepts.difference(ground_truth_concepts)),
        "false_negatives": list(ground_truth_concepts.difference(generated_concepts))
    }
# --- End of NEW: Evaluation Framework Setup ---


app = Flask(__name__)

# Enable CORS
CORS(app)


# --- NEW GUARDRAIL FUNCTION  ---

# ---  UPDATED GUARDRAIL FUNCTION (Fixes HTML Artifacts)  ---
def check_safety_guardrail(generated_text: str, raw_context: str) -> Dict[str, Any]:
    """
    Security Layer: Checks if the LLM generated medical concepts 
    that do not exist in the source text (Hallucination Check).
    Includes robust cleaning to prevent HTML tags from triggering false positives.
    """
    if NLP_MODEL is None:
        return {"status": "DISABLED", "hallucinated_concepts": []}

    # 1. Aggressive HTML Cleaning
    # We replace tags with spaces to prevent words from merging (e.g., "DrugA</li><li>DrugB")
    clean_text = re.sub(r'<[^>]*>', ' ', generated_text)

    # 2. Extract entities using AI (scispaCy)
    generated_ents = get_entities(clean_text)
    context_ents = get_entities(raw_context)

    # 3. Find Potential Hallucinations (AI Set Difference)
    potential_hallucinations = list(generated_ents.difference(context_ents))

    # 4. String Presence Check (The Fallback Fix)
    confirmed_hallucinations = []

    for entity in potential_hallucinations:
        # A. Clean the entity itself (Remove any lingering tags/punctuation)
        # This fixes the "penicillin v</li>" issue
        clean_entity_str = re.sub(r'<[^>]*>', '', entity).strip()

        # B. Check if this clean string exists in the raw context
        # We use case-insensitive matching
        if clean_entity_str and clean_entity_str.lower() not in raw_context.lower():
            confirmed_hallucinations.append(clean_entity_str)

    # 5. Filter out noise (very short words like 'mg', 'tab')
    final_hallucinations = [h for h in confirmed_hallucinations if len(h) > 2]

    if final_hallucinations:
        return {
            "status": "FLAGGED",
            "warning": "Potential Hallucination Detected",
            "message": f"The model mentioned {len(final_hallucinations)} concepts not found in the source records.",
            "hallucinated_concepts": final_hallucinations
        }
    else:
        return {
            "status": "SAFE",
            "message": "No hallucinations detected.",
            "hallucinated_concepts": []
        }


@app.route('/api/rag_query', methods=['POST'])
def handle_rag_query():
    """
    API Gateway Endpoint:
    Executes RAG, Evaluation (F1), and Safety Guardrails.
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        mode = data.get('mode', 'A')

        if not query:
            return jsonify({"error": "Query parameter is required."}), 400

        print(f"Flask Server: Received /api/rag_query. Mode: {mode}")

        # 1. Get RAG Response
        response_data = get_rag_response(query, mode)

        # 2. Run Evaluation (F1 Score)
        if NLP_MODEL:
            response_data['evaluation_metrics'] = calculate_concept_f1(
                response_data['answer'], query)
        else:
            response_data['evaluation_metrics'] = {
                "error": "NLP model not loaded."}

        # 3. Run Safety Guardrail
        # We use the 'raw_context' we extracted in rag_core_service
        raw_ctx = response_data.get('raw_context', '')

        response_data['safety_guardrail'] = check_safety_guardrail(
            response_data['answer'],
            raw_ctx
        )

        # 4. Cleanup (Don't send massive raw text to frontend)
        # response_data.pop('raw_context', None)

        return jsonify(response_data)

    except Exception as e:
        print(f"FATAL API Gateway Error: {e}", file=sys.stderr)
        return jsonify({
            "error": "Internal Server Error",
            "details": str(e),
            "model_name": "System Error"
        }), 500


@app.route('/api/submit_correction', methods=['POST'])
def submit_correction():
    """
    Saves Query, Original Answer, and Corrected Answer in a structured JSON array.
    """
    try:
        data = request.get_json()

        # Validation: Ensure we have all the text fields we need
        required = ['query', 'original_answer', 'corrected_answer']
        if not all(k in data for k in required):
            return jsonify({"error": "Missing required fields"}), 400

        # 1. Create the clean entry
        new_entry = {
            "instruction": data['query'],
            # <--- Added this field
            "original_model_output": data['original_answer'],
            "corrected_output": data['corrected_answer']
        }

        # 2. Read existing data (if file exists)
        existing_data = []
        if os.path.exists(DATASET_FILE):
            try:
                with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        existing_data = json.loads(content)
            except json.JSONDecodeError:
                print(
                    "Warning: Could not decode existing JSON. Starting fresh.", file=sys.stderr)
                existing_data = []

        # 3. Add new entry to the list
        existing_data.append(new_entry)

        # 4. Write back the entire list with Pretty Printing
        with open(DATASET_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

        print(f" Saved correction for: {data['query'][:30]}...")
        return jsonify({"status": "success", "message": "Correction saved successfully"}), 200

    except Exception as e:
        print(f"Error saving correction: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Flask application startup
    print("Flask API Gateway starting...")
    # Run the app locally on host 0.0.0.0 (accessible by network/Docker)
    # Port 8000 now matches the vite.config.ts proxy
    app.run(host='0.0.0.0', port=8000, debug=True)
