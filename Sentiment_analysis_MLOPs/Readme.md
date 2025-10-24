

# 📘 Sentiment Analysis Fullstack App | Day 18 of #90DaysMLChallenge

This project wraps a **Word2Vec + TF-IDF weighted Logistic Regression sentiment model** into a **Flask API**, now integrated with a **React + TailwindCSS + TypeScript frontend** and fully containerized using **Docker**.  

It marks a key milestone — taking machine learning from notebook → API → container → frontend → deploy-ready fullstack application.

---

## Overview
- Trained a custom Word2Vec model on IMDB reviews to learn text embeddings.
- Combined TF-IDF weights to emphasize informative words.
- Built a Logistic Regression classifier for sentiment prediction.
- Deployed backend using Flask for real-time inference.
- Built a **frontend** using React + TypeScript + TailwindCSS for interactive sentiment prediction.
- Containerized the fullstack app with Docker for consistent deployment anywhere.

---

## Tech Stack
- **Backend:** Python 3.12, Flask, scikit-learn, gensim, NumPy, pickle, joblib  
- **Frontend:** React, Vite, TypeScript, TailwindCSS  
- **Containerization:** Docker 🐳  

---

## Project Structure

| File / Folder           | Description                                           |
|-------------------------|-------------------------------------------------------|
| `backend/fast_word2vec.model` | Trained Word2Vec model                              |
| `backend/tfidf_vectorizer.pkl` | TF-IDF vectorizer                                   |
| `backend/classifier.pkl` | Logistic Regression model                              |
| `backend/app.py`        | Flask API file                                       |
| `frontend/`             | React + TailwindCSS frontend code                    |
| `Dockerfile`            | Docker configuration for fullstack containerization |
| `requirements.txt`      | Python dependencies                                  |

---

## Run Locally

### Backend Only
1️⃣ **Setup virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run Flask API

python backend/app.py

API will run at:
👉 http://127.0.0.1:8000

⸻

Frontend

1️⃣ Navigate to frontend folder:

cd frontend

2️⃣ Install dependencies:

npm install

3️⃣ Start development server:

npm run dev

Frontend will run at:
👉 http://localhost:5173
It interacts with your Flask API for live sentiment predictions.

⸻

Docker Setup (Fullstack)

This Dockerfile sets up both backend and frontend in a single container.

Dockerfile Example

# -------------------------------
# Backend
# -------------------------------
FROM python:3.12-slim AS backend

WORKDIR /app/backend

# Copy backend files
COPY backend/ /app/backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Frontend
# -------------------------------
FROM node:20-alpine AS frontend

WORKDIR /app/frontend

# Copy frontend files
COPY frontend/ /app/frontend/

# Install frontend dependencies
RUN npm install
RUN npm run build

# -------------------------------
# Final Image
# -------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy backend
COPY --from=backend /app/backend /app/backend

# Copy frontend build
COPY --from=frontend /app/frontend/dist /app/frontend/dist

# Expose ports
EXPOSE 8000
EXPOSE 5173

# Start backend
CMD ["python", "backend/app.py"]

Build and Run

1️⃣ Build Docker Image

docker build -t sentiment-fullstack .

2️⃣ Run Container

docker run -p 8000:8000 -p 5173:5173 sentiment-fullstack

	•	Backend API: http://127.0.0.1:8000
	•	Frontend: http://localhost:5173

3️⃣ Optional: Push to Docker Hub

docker tag sentiment-fullstack yashbishnoi98/sentiment-fullstack:latest
docker push yashbishnoi98/sentiment-fullstack:latest


⸻

Use Cases

This sentiment analysis fullstack app can be used for:
	•	Product Reviews Analysis – Quickly gauge user sentiment trends.
	•	Social Media Monitoring – Detect public opinion on topics or campaigns.
	•	Chatbots & Customer Support – Identify positive or negative messages automatically.
	•	Content Moderation – Flag negative or abusive text in forums or comments.
	•	Market Research – Aggregate sentiment from multiple sources for business insights.

⸻

Author: Yash Bishnoi
Part of the #90DaysMLChallenge — Building one ML project a day!

#MLOps #Docker #Flask #React #TailwindCSS #TypeScript #MachineLearning #Fullstack #AI #DataScience

