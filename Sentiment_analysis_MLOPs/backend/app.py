# app.py
from flask_cors import CORS
import pickle
import joblib
import re
import numpy as np
from flask import Flask, request, jsonify
from gensim.models import Word2Vec
from flask_cors import CORS

# -----------------------------
#  Load Saved Models
# -----------------------------

# Load Word2Vec model
w2v_model = Word2Vec.load("fast_word2vec.model")

# Load classifier (e.g., LogisticRegression)
classifier = joblib.load("classifier.pkl")

# Load TF-IDF vectorizer
with open("tfidf_vector.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)  # must be TfidfVectorizer object

# -----------------------------
#  Utility Functions
# -----------------------------


def clean_text(text):
    """
    Clean input text by lowercasing and removing non-alphabetic characters.
    Returns a list of tokens.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = text.split()
    return tokens


def vectorize_text(tokens):
    """
    Convert tokens into a TF-IDF weighted Word2Vec vector.
    Returns a single vector for the text.
    """
    vectors = []
    for word in tokens:
        if word in w2v_model.wv:
            # Use TF-IDF weight if the word exists, else weight=1
            if hasattr(tfidf_vectorizer, "vocabulary_") and word in tfidf_vectorizer.vocabulary_:
                weight = tfidf_vectorizer.idf_[
                    tfidf_vectorizer.vocabulary_[word]]
            else:
                weight = 1
            vectors.append(w2v_model.wv[word] * weight)

    if vectors:
        # Compute mean vector for all words
        return np.mean(vectors, axis=0).reshape(1, -1)
    else:
        # If no words matched, return zero vector
        return np.zeros((1, w2v_model.vector_size))


# -----------------------------
#  Create Flask App
# -----------------------------

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "Flask API for Sentiment Analysis is running!"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for input text.
    Expects JSON: {"text": "Your text here"}
    """
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    tokens = clean_text(text)
    vector = vectorize_text(tokens)

    # Predict sentiment
    prediction = classifier.predict(vector)[0]
    sentiment = "positive" if prediction == 1 else "negative"

    return jsonify({
        "text": text,
        "predicted_sentiment": sentiment
    })


# -----------------------------
#  Run the Flask API
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
