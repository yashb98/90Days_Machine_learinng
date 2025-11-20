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
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialize Engines (This loads the general English model)
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

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


def clean_text_artifacts(text: str) -> str:
    """
    1. Removes trailing numbers from words (e.g., "Donetta1" -> "Donetta").
    2. Removes excessive whitespace.
    """
    # Regex: Find words that end in digits, keep only the word part
    # pattern: ([a-zA-Z]+)\d+ -> replace with \1 (the letters only)
    text_clean = re.sub(r'\b([a-zA-Z]+)\d+\b', r'\1', text)

    # Optional: Clean up extra spaces caused by removals
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    return text_clean


def redact_pii(text: str) -> str:
    """
    Detects and replaces PII (Names, Dates, IDs) with placeholders like <PERSON>.
    Uses Microsoft Presidio for industry-standard de-identification.
    """
    try:
        # 1. Analyze: Find the PII entities
        # We explicitly look for PERSON, LOCATION, DATE_TIME, etc.
        results = analyzer.analyze(
            text=text,
            entities=["PERSON", "PHONE_NUMBER",
                      "EMAIL_ADDRESS", "DATE_TIME", "LOCATION"],
            language='en'
        )

        # 2. Anonymize: Replace them with tags
        anonymized_result = anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )

        return anonymized_result.text
    except Exception as e:
        print(f"PII Redaction Warning: {e}", file=sys.stderr)
        return text  # Fallback: return original text if scrubber fails


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


# --- GUARDRAIL FUNCTION  ---


def check_safety_guardrail(generated_text: str, raw_context: str) -> Dict[str, Any]:
    """
    Security Layer: Checks if the LLM generated medical concepts 
    that do not exist in the source text (Hallucination Check).
    Handles HTML, Markdown, and Case Sensitivity.
    """
    if NLP_MODEL is None:
        return {"status": "DISABLED", "hallucinated_concepts": []}

    # 1. Clean HTML tags (replace with space to avoid word merging)
    # e.g. "DrugA</li><li>DrugB" -> "DrugA  DrugB"
    text_no_html = re.sub(r'<[^>]+>', ' ', generated_text)

    # 2. Clean Markdown symbols (replace with space)
    # e.g. "**Allergies**Penicillin" -> "  Allergies  Penicillin"
    clean_text = re.sub(r'[\*\#\-\•]', ' ', text_no_html)

    # 3. Extract entities using AI (scispaCy)
    # Note: get_entities() already converts everything to .lower()
    generated_ents = get_entities(clean_text)
    context_ents = get_entities(raw_context)

    # 4. Find Potential Hallucinations (AI Set Difference)
    potential_hallucinations = list(generated_ents.difference(context_ents))

    # 5. String Presence Check (The Ultimate Fallback)
    confirmed_hallucinations = []

    # We normalize the context once for speed and accuracy
    # Remove punctuation from context to ensure "Penicillin-V" matches "Penicillin V"
    normalized_context = re.sub(r'[^\w\s]', ' ', raw_context).lower()

    for entity in potential_hallucinations:
        # Clean the entity string (remove punctuation, trim spaces)
        clean_entity_str = re.sub(r'[^\w\s]', ' ', entity).strip().lower()

        # Check if this clean string exists in the normalized context
        if clean_entity_str and clean_entity_str not in normalized_context:
            confirmed_hallucinations.append(entity)

    # 6. Filter out noise (very short words like 'mg', 'no', 'dr')
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
    Robustly handles file corruption or format mismatches.
    """
    try:
        data = request.get_json()

        # Validation
        required = ['query', 'original_answer', 'corrected_answer']
        if not all(k in data for k in required):
            return jsonify({"error": "Missing required fields"}), 400

        # 1. Clean Artifacts (Donetta1 -> Donetta)
        # We clean the text first so standard AI models can recognize the names
        q_clean = clean_text_artifacts(data['query'])
        a_orig_clean = clean_text_artifacts(data['original_answer'])
        a_corr_clean = clean_text_artifacts(data['corrected_answer'])

        clean_instruction = redact_pii(q_clean)
        clean_original = redact_pii(a_orig_clean)
        clean_correction = redact_pii(a_corr_clean)

        new_entry = {
            "instruction": clean_instruction,
            "original_model_output": clean_original,
            "corrected_output": clean_correction
        }

        existing_data = []

        # Read existing data
        if os.path.exists(DATASET_FILE):
            try:
                with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        loaded_data = json.loads(content)

                        # --- FIX: Ensure it's a list ---
                        if isinstance(loaded_data, list):
                            existing_data = loaded_data
                        elif isinstance(loaded_data, dict):
                            # If it's a single dict, wrap it in a list
                            print(
                                "Warning: Converting dict dataset to list.", file=sys.stderr)
                            existing_data = [loaded_data]
                        else:
                            # Unknown format, start fresh
                            print(
                                "Warning: Dataset format unknown. Starting fresh.", file=sys.stderr)
                            existing_data = []

            except json.JSONDecodeError:
                print(
                    "Warning: Could not decode existing JSON. Starting fresh.", file=sys.stderr)
                existing_data = []

        # Add new entry
        existing_data.append(new_entry)

        # Write back
        with open(DATASET_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

        print(f" Saved correction for: {data['query'][:30]}...")
        return jsonify({"status": "success", "message": "Correction saved successfully"}), 200

    except Exception as e:
        print(f"Error saving correction: {e}", file=sys.stderr)
        # Return the specific error message to the frontend for easier debugging
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Flask application startup
    print("Flask API Gateway starting...")
    # Run the app locally on host 0.0.0.0 (accessible by network/Docker)
    # Port 8000 now matches the vite.config.ts proxy
    app.run(host='0.0.0.0', port=8000, debug=True)
