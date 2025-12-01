# 👁️ Project Aura: Intelligent Seeing Companion

**A real-time, multimodal AI assistant for the visually impaired.**

Project Aura turns a smartphone into a verbal guide. It streams live video from the user's camera to Google's Gemini Live API, processes visual data in real-time, and streams back instant audio descriptions of obstacles, text, and environmental context.

---

## 🏗️ Architecture & Tech Stack

### **The "Eyes" (Mobile Frontend)**
* **Framework:** Flutter (Dart)
* **Camera:** `camera` package (Raw YUV420 streaming)
* **Audio:** `sound_stream` (Raw PCM 24kHz playback)
* **Protocol:** Secure WebSockets (`wss://`)
* **Compute:** Dart Isolates (Background thread image compression)

### **The "Brain" (Cloud Backend)**
* **Language:** Python 3.11
* **Framework:** FastAPI (Async WebSocket Server)
* **AI Model:** Gemini 2.0 Flash Experimental (Multimodal Live API)
* **Containerization:** Docker (Slim image)
* **Hosting:** Google Cloud Run (Serverless, Auto-scaling)
* **Security:** Firebase Authentication & Google Secret Manager

---

## 📅 Build Log: From Zero to Cloud

### **Phase 1: Foundation (Days 46-49)**

#### **Day 46: Infrastructure & Cloud Setup**
* **Goal:** Lay the groundwork for a cloud-native application.
* **Achievements:**
    * Initialized Google Cloud Project (`aura-backend-project`).
    * Configured **Artifact Registry** for secure Docker image storage.
    * Created dedicated Service Accounts with least-privilege access.
    * Built the initial `Dockerfile` for the Python backend.

#### **Day 47: The "Eyes" (Raw Vision)**
* **Goal:** Get raw visual data out of the smartphone hardware.
* **Achievements:**
    * Initialized the Flutter mobile app.
    * Bypassed standard photo capture to access the **Image Stream**.
    * Successfully extracted **YUV420 / NV21** raw byte planes from the camera sensor.

#### **Day 48: Throttling & Optimization**
* **Goal:** Prevent the "firehose" of data from crashing the network.
* **Achievements:**
    * **Throttling:** Implemented logic to cap transmission at ~1 frame per 1.5 seconds.
    * **Isolates:** Moved heavy image compression to a background thread (`compute`) to prevent UI freeze.
    * **Compression:** Resized frames to VGA (640x480) and compressed to JPEG (Quality 50%) to minimize payload size (~40KB).

#### **Day 49: The "Voice" (Audio Engine)**
* **Goal:** Enable instant, low-latency audio feedback.
* **Achievements:**
    * Rejected standard MP3 players due to buffering latency.
    * Implemented a **Raw PCM Player** using `sound_stream`.
    * Configured the audio sink to accept **24kHz, 16-bit, Mono** chunks directly from the socket.

---

### **Phase 2: Intelligence & Connection (Days 50-52)**

#### **Day 50: The Streaming Pipeline**
* **Goal:** Connect the phone to the brain.
* **Achievements:**
    * Built an Async **FastAPI WebSocket Server**.
    * Implemented a **Proxy Architecture**:
        * *Upstream:* Mobile -> Backend (Images).
        * *Downstream:* Backend -> Mobile (Audio).
        * *AI Link:* Backend -> Gemini Live API (Server-to-Server).
    * Successfully streamed video to Gemini and received audio response chunks.

#### **Day 51: Security & UI**
* **Goal:** Secure the API and polish the user experience.
* **Achievements:**
    * **Firebase Auth:** Implemented anonymous login and ID Token generation on mobile.
    * **Token Verification:** Backend now rejects any WebSocket connection without a valid, unexpired Firebase token.
    * **UI:** Built a modern, glass-morphism Login Screen.

#### **Day 52: Cloud Deployment (The Big Milestone)**
* **Goal:** Cut the cord. Make the app work on 4G/5G.
* **Achievements:**
    * **Docker Optimization:** Fixed critical startup crashes by switching to `sh -c` command format to handle Cloud Run's dynamic `$PORT`.
    * **Secret Management:** Injected API Keys via Google Secret Manager (no hardcoded secrets).
    * **Deploy:** Successfully deployed to **Google Cloud Run**.
    * **Result:** App connects securely via `wss://aura-backend-service...run.app`.

---

### **Phase 3: Intelligence Refinement (Day 53+)**

#### **Day 53: Persona & Prompt Engineering**
* **Goal:** Stop the AI from being a chatbot; make it a safety guide.
* **Achievements:**
    * Defined the **System Instruction**: "You are Aura, a safety-oriented navigation assistant..."
    * Established priority logic: **Dangers > Text > Scenery**.
    * Tuned for brevity (max 2 sentences, no filler words).

---

## 🚀 How to Run Locally

### **Prerequisites**
* Flutter SDK
* Python 3.11
* Docker
* Firebase Project & `serviceAccountKey.json`

### **1. Backend (Local)**

cd backend
# Install dependencies
pip install -r requirements.txt
# Run server
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

### **2. Mobile App

Connect physical Android device via USB.

Update lib/main.dart to use local IP (e.g., ws://192.168.1.5:8080/ws).

cd aura_app
flutter run


###☁️ How to Deploy to Cloud

**1. Build Docker Image

export PROJECT_ID=$(gcloud config get-value project)
gcloud builds submit --tag europe-west2-docker.pkg.dev/$PROJECT_ID/aura-backend/api:v1


**2. Deploy to Cloud Run

gcloud run deploy aura-backend-service \
  --image=europe-west2-docker.pkg.dev/$PROJECT_ID/aura-backend/api:v1 \
  --region=europe-west2 \
  --allow-unauthenticated \
  --port=8080 \
  --timeout=300 \
  --set-secrets="GEMINI_API_KEY=gemini-api-key:latest" \
  --set-secrets="/app/serviceAccountKey.json=firebase-service-account:latest"


###📂 Project Structure

Aura/
├── backend/                 # Python FastAPI Brain
│   ├── Dockerfile           # Production container config
│   ├── main.py              # WebSocket Proxy & Gemini Logic
│   ├── requirements.txt     # Dependencies
│   └── serviceAccountKey.json # (Ignored) Firebase Admin Creds
│
└── aura_app/                # Flutter Mobile App
    ├── lib/
    │   ├── main.dart        # Auth Routing & Entry
    │   ├── screens/
    │   │   ├── login_screen.dart  # Auth UI
    │   └── services/
    │       └── audio_player_service.dart # PCM Audio Engine
    └── pubspec.yaml
