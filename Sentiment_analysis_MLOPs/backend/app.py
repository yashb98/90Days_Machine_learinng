from flask_cors import CORS
import pickle
import joblib
import re
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from gensim.models import Word2Vec
import os
import sys


# -----------------------------
# Load Saved Models (Fixed Path)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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
            raise RuntimeError(f"Failed to load model {path}: {e}")


try:
    w2v_model = Word2Vec.load(os.path.join(BASE_DIR, "fast_word2vec.model"))
    classifier = joblib.load(os.path.join(BASE_DIR, "classifier.pkl"))
    tfidf_vectorizer = load_model_safely(
        os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")


# -----------------------------
# Utility Functions
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
app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    tokens = clean_text(text)
    vector = vectorize_text(tokens)
    prediction = classifier.predict(vector)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return jsonify({"text": text, "predicted_sentiment": sentiment})

# Serve React frontend


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
