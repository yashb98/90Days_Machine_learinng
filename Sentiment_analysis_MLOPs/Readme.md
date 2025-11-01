
⸻

 CI/CD for ML: Dockerized Sentiment Analysis Fullstack App

Day 26 of #90DaysMLChallenge — Continuous Integration & Deployment (CI/CD)

This project extends the Sentiment Analysis Fullstack App from Day 18 by introducing CI/CD automation — enabling continuous integration, testing, containerization, and deployment pipelines for machine learning models.

It demonstrates how to move from local ML experiments to production-ready, automated MLOps pipelines.

⸻

🧠 Overview
	•	Built a Sentiment Analysis App using a Word2Vec + TF-IDF + Logistic Regression pipeline.
	•	Served the ML model via a Flask REST API.
	•	Integrated a React + TailwindCSS + TypeScript frontend.
	•	Containerized both frontend and backend in a single multi-stage Docker image.
	•	Set up CI/CD workflows for automated build, test, and deployment using GitHub Actions (or GitLab CI/Jenkins).

⸻

⚙️ Tech Stack

Layer	Technology	Purpose
ML Model	Word2Vec, TF-IDF, Logistic Regression	Text vectorization and sentiment classification
Backend	Python 3.11, Flask, Flask-CORS	Serve ML predictions via REST API
Frontend	React, TypeScript, TailwindCSS, Vite	Modern, fast UI for user sentiment input
Containerization	Docker	Unified fullstack image
CI/CD	GitHub Actions	Automated testing, build, and deployment
Deployment	Docker Hub / AWS / Render / Railway	Production-ready hosting


⸻

📁 Project Structure

📦 Sentiment-Fullstack-CICD
├── backend/
│   ├── app.py
│   ├── classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── fast_word2vec.model
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       └── components/
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci-cd.yml
└── README.md


⸻

🧩 Dockerfile (Multi-Stage Build)

# -----------------------------
# Stage 1: Build Frontend
# -----------------------------
FROM node:20 AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# -----------------------------
# Stage 2: Backend + Serve Frontend
# -----------------------------
FROM python:3.11-slim
WORKDIR /app
COPY backend/ ./backend
WORKDIR /app/backend
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flask-cors gensim numpy joblib
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist
EXPOSE 5002
CMD ["python", "app.py"]


⸻

🔄 CI/CD Workflow (GitHub Actions)

Create this file at:
.github/workflows/ci-cd.yml

name: CI/CD Pipeline

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Backend Dependencies
        run: |
          pip install --upgrade pip
          pip install -r backend/requirements.txt

      - name: Run Backend Tests
        run: |
          echo " Running backend test step (add pytest or curl-based tests here)"

      - name: Build Frontend
        uses: actions/setup-node@v4
        with:
          node-version: "20"
        run: |
          cd frontend
          npm install
          npm run build

      - name: Build Docker Image
        run: |
          docker build -t yashbishnoi/sentiment-fullstack:latest .

      - name: Push to Docker Hub
        run: |
          echo "${{ secrets.DOCKERHUB_TOKEN }}" | docker login -u "yashbishnoi" --password-stdin
          docker push yashbishnoi/sentiment-fullstack:latest

💡 This workflow automatically builds, tests, and deploys your Dockerized ML app to Docker Hub whenever you push to main.

⸻

🧭 Running Locally

1. Clone Repo

git clone https://github.com/yashb98/90Days_Machine_learinng.git
cd 90Days_Machine_learinng/Sentiment_analysis_MLOPs

2. Build & Run Docker

docker build -t sentiment-fullstack .
docker run -p 5002:5002 sentiment-fullstack

	•	Backend → http://localhost:5002
	•	Frontend (React) → http://localhost:5173

⸻

🚢 Deployment Options

Platform	Deployment Method
Docker Hub	Push via GitHub Actions
Render / Railway	Deploy full Docker container
AWS ECS / EC2	Run containerized Flask+React app
GitHub Pages (frontend)	Serve frontend separately


⸻

🧠 Learning Outcomes

1. CI/CD Pipeline design for ML projects
2. Multi-stage Docker build for fullstack apps
3. Integration of React + Flask + ML model
4. Automation using GitHub Actions
5. Containerized deployment best practices

⸻

👨‍💻 Author

Yash Bishnoi
University of Dundee | MSc Computer Science
Part of the #90DaysMLChallenge — Building an ML project every day

📫 Connect: LinkedIn ( https://www.linkedin.com/in/yash-bishnoi-2ab36a1a5/ ) 

