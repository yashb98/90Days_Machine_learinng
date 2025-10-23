Perfect! Here’s the updated README.md with a Use Cases section added. You can replace your current README with this.

# 📘 Sentiment Analysis API with Flask | Days 16–17 of #90DaysMLChallenge

This project wraps a Word2Vec + TF-IDF weighted Logistic Regression sentiment model into a Flask API, now fully containerized using Docker .  

It marks a key milestone — taking machine learning from notebook → API → container → deploy-ready application. 

---

##  Overview
- Trained a custom Word2Vec model on IMDB reviews to learn text embeddings.
- Combined TF-IDF weights to emphasize informative words.
- Built a Logistic Regression classifier for sentiment prediction.
- Deployed using Flask for real-time inference.
- Containerized the full app using Docker for consistent deployment anywhere.

---

##  Tech Stack
- Python 3.12
- Flask (API backend)
- scikit-learn (ML model)
- gensim (Word2Vec embeddings)
- NumPy, pickle, joblib
- Docker 🐳 (Containerization)

---

##  Project Structure

| File                    | Description                            |
|-------------------------|----------------------------------------|
| fast_word2vec.model     | Trained Word2Vec model                  |
| tfidf_vectorizer.pkl    | TF-IDF vectorizer                        |
| classifier.pkl          | Logistic Regression model               |
| app.py                  | Flask API file                          |
| Dockerfile              | Docker configuration for containerization |
| requirements.txt        | Python dependencies                     |

---

##  Run Locally

1️⃣ **Setup virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run Flask API

python app.py

Your API will run at:
👉 http://127.0.0.1:8000

⸻

 Test API

Using curl or Postman:

curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "I absolutely loved this movie!"}'

 Expected Output:

{
  "text": "I absolutely loved this movie!",
  "predicted_sentiment": "positive"
}


⸻

 Using Docker (Step-by-Step)

If you have Docker installed, you can run the API without installing Python or dependencies.

1️⃣ Build Docker Image

docker build -t sentiment-flask-app .

2️⃣ Verify Image

docker images

You should see:

REPOSITORY              TAG       IMAGE ID       CREATED          SIZE
sentiment-flask-app     latest    123abc456def   2 minutes ago    650MB

3️⃣ Run the Container

docker run -p 8000:8000 sentiment-flask-app

API is now live inside Docker:
👉 http://127.0.0.1:8000

4️⃣ Test the API (Inside Docker)

curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "This film was terrible!"}'

Expected Response:

{
  "text": "This film was terrible!",
  "predicted_sentiment": "negative"
}

5️⃣ (Optional) Push to Docker Hub

docker tag sentiment-flask-app yashbishnoi98/sentiment-flask-app:latest
docker push yashbishnoi98/sentiment-flask-app:latest


⸻

 Use Cases

This sentiment analysis API can be used for:
	•	Product Reviews Analysis – Analyze user reviews to quickly gauge sentiment trends.
	•	Social Media Monitoring – Detect public opinion on topics, hashtags, or campaigns.
	•	Chatbots & Customer Support – Automatically identify positive or negative messages for better response prioritization.
	•	Content Moderation – Flag negative or abusive text in forums, comments, or feedback systems.
	•	Market Research – Aggregate sentiment from multiple sources to inform business decisions.

⸻

What’s Next (Day 18)

 Add a React + Tailwind CSS frontend to interact with this API visually — making sentiment predictions more intuitive and user-friendly.

⸻

Author: Yash Bishnoi

Part of the #90DaysMLChallenge — Building one ML project a day!

#MLOps #Docker #Flask #MachineLearning #DevOps #AI #DataScience
