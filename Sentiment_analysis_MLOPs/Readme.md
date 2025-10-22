Perfect ✅ — here’s a clean and engaging README.md for your
Sentiment_analysis_MLOPs folder, written at an intermediate ML engineer level.
It summarizes your work clearly and looks professional on GitHub.

⸻

📘 README.md

# 🎯 Sentiment Analysis API with Flask | Day 16 of #90DaysMLChallenge

This project wraps a **Word2Vec + TF-IDF weighted Logistic Regression sentiment model** into a **Flask API**, marking the first step into **MLOps** — taking machine learning from notebooks to deployable applications. 🚀

---

## 🧠 Overview
We trained a custom **Word2Vec model** on IMDB movie reviews to generate semantic embeddings.  
These embeddings were weighted using **TF-IDF scores** to emphasize informative words.  
A **Logistic Regression classifier** was then trained on these features to predict sentiment (Positive/Negative).  
Finally, the model was deployed as a Flask API to handle real-time text predictions.

---

## ⚙️ Tech Stack
- **Python 3.12**
- **Flask** (for API serving)
- **scikit-learn** (Logistic Regression, TF-IDF)
- **gensim** (Word2Vec embeddings)
- **NumPy / pickle / joblib**

---

## 🧩 Model Files
| File | Description |
|------|--------------|
| `fast_word2vec.model` | Trained Word2Vec model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `classifier.pkl` | Trained Logistic Regression classifier |
| `app.py` | Flask app to serve the model |

---

## 🚀 Run the API
1. **Create & activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

	2.	Install dependencies

pip install -r requirements.txt


	3.	Run Flask API

python app.py


	4.	Test using cURL

curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "I absolutely loved this movie!"}'



Expected Response:

{
  "text": "I absolutely loved this movie!",
  "predicted_sentiment": "positive"
}


⸻

🧭 What’s Next (Day 17)

🔹 Containerize this Flask app using Docker
🔹 Containerizing my ML application with Docker. This ensures it runs the same everywhere, from my laptop to the cloud.

⸻

Author: Yash Bishnoi
 Part of the #90DaysMLChallenge — Building one ML project a day!
