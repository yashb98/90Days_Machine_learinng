from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv  # 1. Utility to load environment variables from .env
import os
import sys  # Added for path correction
from typing import Dict, Any, Set
import spacy
# import medspacy # Not directly used but good to know it's there

# --- Load Environment Variables from .env File (MUST BE FIRST) ---
load_dotenv()

# --- Path Setup ---
# This ensures we can import from 'rag_core_service.py'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
# --- End Path Setup ---

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


@app.route('/api/rag_query', methods=['POST'])
def handle_rag_query():
    """
    API Gateway Endpoint:
    Receives query, calls RAG, and now also calculates F1 score.
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        mode = data.get('mode', 'A')

        if not query:
            return jsonify({"error": "Query parameter is required."}), 400

        print(f"Flask Server: Received /api/rag_query. Mode: {mode}")

        # Call the RAG Core Service to execute the retrieval and LLM call
        response_data = get_rag_response(query, mode)

        # --- NEW: Calculate F1 Score ---
        if NLP_MODEL:
            # We use the raw query as the key for our ground truth DB
            eval_metrics = calculate_concept_f1(response_data['answer'], query)
            # Add the metrics to our JSON response
            response_data['evaluation_metrics'] = eval_metrics
        else:
            response_data['evaluation_metrics'] = {
                "error": "NLP model not loaded."}
        # --- End of NEW: Calculate F1 Score ---

        return jsonify(response_data)

    except Exception as e:
        # Critical error logging
        print(f"FATAL API Gateway Error: {e}", file=sys.stderr)
        return jsonify({
            "error": "Internal Server Error during RAG execution.",
            "details": str(e),  # <-- UPDATED: Provide specific error details
            "model_name": "System Error"
        }), 500


if __name__ == '__main__':
    # Flask application startup
    print("Flask API Gateway starting...")
    # Run the app locally on host 0.0.0.0 (accessible by network/Docker)
    # Port 8000 now matches the vite.config.ts proxy
    app.run(host='0.0.0.0', port=8000, debug=True)
