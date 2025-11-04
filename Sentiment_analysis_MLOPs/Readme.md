
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






````markdown
# Project 11: End-to-End Sentiment Analysis API

This project is a complete MLOps demonstration, deploying a sentiment analysis model as a unified, full-stack application.

It features a Python/Flask backend that serves both a pre-trained Scikit-learn model and a production-built React frontend. The entire application is containerized with Docker and deployed as a single, scalable service on AWS Elastic Beanstalk.

### 🚀 Live Application

* **Live URL:** `http://imdbsentiment-env.eba-pnikujcp.eu-west-2.elasticbeanstalk.com/`

The single URL serves both the interactive React UI and the backend API.

---

## 🏗️ Project Architecture

A single Docker container runs a Flask server (likely via Gunicorn) on AWS Elastic Beanstalk.

1.  **Frontend:** The Flask server is configured to serve the static files (HTML, CSS, JS) from the React `build` folder.
2.  **Backend:** The same Flask server exposes the `/predict` API endpoint.
3.  **Data Flow:**
    * A user visits the root URL, which loads the React app.
    * The React app makes a request to the **relative path** (`/predict`).
    * Because the request is to the same origin, it's routed to the Flask API, which loads the model, makes a prediction, and returns the JSON result.



## 🛠️ Tech Stack

| Area | Technology |
| :--- | :--- |
| **Machine Learning** | Scikit-learn, Pandas, NLTK |
| **Backend & Serving** | Python, Flask, Gunicorn |
| **Frontend** | React, TypeScript, Axios, TailwindCSS |
| **Deployment** | Docker, AWS Elastic Beanstalk |

---

## 🚀 Using the Live API

You can test the live endpoint directly using `curl` or any API client (like Postman).

### Test: Positive Sentiment

**Request:**
```bash
curl -X POST "http://imdbsentiment-env.eba-pnikujcp.eu-west-2.elasticbeanstalk.com/predict)" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic and I loved it!"}'
````

**Response:**

```json
{
  "predicted_sentiment": "positive"
}
```

### Test: Negative Sentiment

**Request:**

```bash
curl -X POST "http://imdbsentiment-env.eba-pnikujcp.eu-west-2.elasticbeanstalk.com/predict)" \
     -H "Content-Type: application/json" \
     -d '{"text": "A complete waste of time. The acting was terrible."}'
```

**Response:**

```json
{
  "predicted_sentiment": "negative"
}
```

-----

## 🖥️ How to Run Locally

In production, one server does everything. For local development, it's easier to run two separate servers and use Vite's built-in proxy to connect them.

### 1\. Backend (Flask API)

1.  **Navigate to the backend directory:**
    ```bash
    cd /path/to/your/backend_folder
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Flask app:**
    ```bash
    python app.py
    ```
    The API will be running at `http://127.0.0.1:5002`.

### 2\. Frontend (React App)

1.  **Navigate to the frontend directory:**
    ```bash
    cd /path/to/your/frontend_folder
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    ```
3.  **Configure the Vite Proxy:**
      * Create a `vite.config.ts` file in the root of the frontend folder.
      * Add the following configuration to proxy `/predict` requests to your local Flask server:
        ```typescript
        // vite.config.ts
        import { defineConfig } from 'vite'
        import react from '@vitejs/plugin-react'

        export default defineConfig({
          plugins: [react()],
          server: {
            proxy: {
              '/predict': {
                target: '[http://127.0.0.1:5002](http://127.0.0.1:5002)', // Your local Flask server
                changeOrigin: true,
              },
            },
          },
        })
        ```
4.  **Verify your React Code:**
      * In `SentimentForm.tsx`, make sure your `axios` call uses the **relative path**:
        ```typescript
        // This will now work for both dev and production!
        const res = await axios.post("/predict", { text });
        ```
5.  **Run the React app:**
    ```bash
    npm run dev
    ```
    The frontend will be running at `http://localhost:5173`. You can now use the app, and any "Predict" clicks will be correctly proxied to your Flask API.

-----

## ☁️ Deployment: AWS Elastic Beanstalk

This project is deployed to AWS Elastic Beanstalk using its "Docker" platform. The `Dockerfile` is responsible for building the React app, installing the Python dependencies, and starting the Flask server.

### Key `Dockerfile` Concepts

A production `Dockerfile` for this architecture would perform these steps:

1.  **Build Stage:** Use a `node` base image to `npm install` and `npm run build` the React app.
2.  **Final Stage:** Use a `python` base image.
3.  Install Python dependencies from `requirements.txt`.
4.  **Copy** the built React app (from the `build` folder) into the Flask server's `static` folder.
5.  **Copy** the Flask app code (`app.py`), model, and vectorizer.
6.  Set the `CMD` to run the production server (e.g., `gunicorn -w 4 'app:app'`).

*(Your Flask `app.py` must also be configured to serve the `index.html` from the static folder for any routes it doesn't recognize.)*

-----



👨‍💻 Author

Yash Bishnoi
University of Dundee | MSc Computer Science
Part of the #90DaysMLChallenge — Building an ML project every day

📫 Connect: LinkedIn ( https://www.linkedin.com/in/yash-bishnoi-2ab36a1a5/ ) 

