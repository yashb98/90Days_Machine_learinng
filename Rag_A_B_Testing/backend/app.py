from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv  # 1. Utility to load environment variables from .env
import os
import sys  # Added for path correction
from typing import Dict, Any

# --- Load Environment Variables from .env File (MUST BE FIRST) ---
load_dotenv()

# --- Path Setup ---
# This ensures we can import from 'rag_core_logic.py'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
# --- End Path Setup ---

# 2. Now import the RAG Core Service.
try:
    # --- CORRECTED IMPORT ---
    # Changed to 'rag_core_logic' to match your file name
    from rag_core_service import get_rag_response
except ImportError as e:
    print("-------------------------------------------------------", file=sys.stderr)
    # --- Updated Error Message ---
    print(f"FATAL ERROR: Could not import from 'rag_core_service.py'.", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    print("Ensure 'app.py' and 'rag_core_service.py' are in the same directory.", file=sys.stderr)
    print("-------------------------------------------------------", file=sys.stderr)
    sys.exit(1)


app = Flask(__name__)

# Enable CORS
CORS(app)


@app.route('/api/rag_query', methods=['POST'])  # <-- ADDED '/api' back
def handle_rag_query():
    """
    API Gateway Endpoint:
    Receives the query and A/B test mode from the React Frontend.
    This route now matches the proxy setup.
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        mode = data.get('mode', 'A')

        if not query:
            return jsonify({"error": "Query parameter is required."}), 400

        # <-- Updated print
        print(f"Flask Server: Received /api/rag_query. Mode: {mode}")

        # Call the RAG Core Service to execute the retrieval and LLM call
        response_data = get_rag_response(query, mode)

        return jsonify(response_data)

    except Exception as e:
        # Critical error logging
        print(f"FATAL API Gateway Error: {e}", file=sys.stderr)
        return jsonify({
            "error": "Internal Server Error during RAG execution.",
            "details": "Check RAG Core Service logs. Ensure GEMINI_API_KEY is set.",
            "model_name": "System Error"
        }), 500


if __name__ == '__main__':
    # Flask application startup
    print("Flask API Gateway starting...")
    # Run the app locally on host 0.0.0.0 (accessible by network/Docker)
    # Port 8000 now matches the vite.config.ts proxy
    app.run(host='0.0.0.0', port=8000, debug=True)
