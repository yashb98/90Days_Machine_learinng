from flask import Flask, request, jsonify
import logging
from flask_cors import CORS
import pickle
import joblib
import re
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from gensim.models import Word2Vec
import os
import sys

# --- 1. SET UP THE LOGGER ---
# We configure a logger to write to standard output (sys.stdout)
# This is what Docker and Elastic Beanstalk will capture.
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,  # Log INFO, WARNING, ERROR, and CRITICAL messages
    # A nice, clean log format
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
# Get a logger instance specifically for our app
logger = logging.getLogger("sentiment-app")


# -----------------------------
# Load Saved Models (Fixed Path)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "backend")


def identity_tokenizer(text):
    """Tokenizer use during TF-IDF training"""
    return text


sys.modules['__main__'].identity_tokenizer = identity_tokenizer
sys.modules['backend.app'] = sys.modules[__name__]


def load_model_safely(path):
    """Try both pickle and joblib for compatibility"""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            return joblib.load(path)
        except Exception as e:
            # --- MODIFIED: Added logging to the error ---
            logger.error(f"Failed to load model {path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model {path}: {e}")


try:
    # --- ADDED: Logging for model loading ---
    logger.info("Loading models...")
    logger.info(f"Loading Word2Vec model from: {MODEL_DIR}")
    w2v_model = Word2Vec.load(os.path.join(
        MODEL_DIR, "fast_word2vec.model"))

    logger.info(f"Loading classifier from: {MODEL_DIR}")
    classifier = joblib.load(os.path.join(
        MODEL_DIR,  "classifier.pkl"))

    logger.info(f"Loading TF-IDF vectorizer from: {MODEL_DIR}")
    tfidf_vectorizer = load_model_safely(
        os.path.join(MODEL_DIR,  "tfidf_vectorizer.pkl"))

    logger.info("All models loaded successfully.")
    # --- END OF ADDED LOGGING ---

except Exception as e:
    # --- MODIFIED: Added logging to the fatal error ---
    logger.error(f"FATAL: Error loading models: {e}", exc_info=True)
    raise RuntimeError(f"Error loading models: {e}")


# -----------------------------
# Utility Functions
# (Your code - unchanged)
# -----------------------------


def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return text.split()


def vectorize_text(tokens):
    vectors = []
    for word in tokens:
        if word in w2v_model.wv:
            weight = tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[
                word]] if word in tfidf_vectorizer.vocabulary_ else 1
            vectors.append(w2v_model.wv[word] * weight)
    return np.mean(vectors, axis=0).reshape(1, -1) if vectors else np.zeros((1, w2v_model.vector_size))


# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__, static_folder=os.path.join(
    BASE_DIR, "frontend", "dist"), static_url_path="")
CORS(app)


# --- MODIFIED: Your /predict endpoint, now with logging and error handling ---
@app.route('/predict', methods=['POST'])
def predict():
    # ADDED: A try/except block for reliability.
    # This prevents one bad request from crashing your whole server.
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")

        if not text:
            # ADDED: Log a warning for bad requests
            logger.warning(f"Invalid request: No text provided. Data: {data}")
            return jsonify({"error": "No text provided"}), 400

        # ADDED: Log the incoming request
        # We truncate to 150 chars to avoid flooding logs with huge text.
        logger.info(f"Received prediction request for text: '{text[:150]}...'")

        # --- YOUR CORE LOGIC (unchanged) ---
        tokens = clean_text(text)
        vector = vectorize_text(tokens)
        prediction = classifier.predict(vector)[0]
        sentiment = "positive" if prediction == 1 else "negative"
        # --- END OF YOUR CORE LOGIC ---

        # ADDED: Log the final model prediction
        logger.info(
            f"Model prediction: {sentiment} (Raw: {prediction}) for text: '{text[:150]}...'")

        return jsonify({"text": text, "predicted_sentiment": sentiment})

    # ADDED: Catch-all exception for any unexpected errors in your code
    except Exception as e:
        # This will log the *full Python error* to CloudWatch for debugging
        logger.error(
            f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500


# Serve React frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    # --- MODIFIED: Added logging to your frontend server ---
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        # Use a "debug" level log so it's not too noisy in production
        logger.debug(f"Serving static file: {path}")
        return send_from_directory(app.static_folder, path)
    else:
        # Log when we serve the main React app
        logger.info(
            f"Serving frontend entrypoint: index.html (path: '{path}')")
        return send_from_directory(app.static_folder, "index.html")


# --- MODIFIED: Your __main__ block, now ready for production ---
if __name__ == "__main__":
    # MODIFIED: Get port from environment variable for Elastic Beanstalk
    port = int(os.environ.get("PORT", 5002))
    # MODIFIED: Set debug=False. debug=True MUST NOT be used in production.
    app.run(host="0.0.0.0", port=port, debug=False)
