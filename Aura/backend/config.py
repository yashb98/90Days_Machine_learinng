# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Toggle this to switch between AI Studio and Vertex AI
# USE_VERTEX_AI = True

# # Vertex AI Settings
# GOOGLE_CLOUD_PROJECT = "aura-backend-project"
# GOOGLE_CLOUD_LOCATION = "us-central1"  # or europe-west2, etc.

# # Keep this for fallback or mixed usage
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_MODEL = "gemini-live-2.5-flash-native-audio"

# # --- BRAIN MODES (System Instructions) ---
# PERSONAS = {
#     "safety": """
#     You are Aura, a navigation guide for blind users.
#     PRIORITY: Accurate, conservative safety info.

#     RULES:
#     1. Never invent or guess dangers. If you are not sure, say "No clear obstacle visible."
#     2. Use [CRITICAL] only when you clearly see one of: moving car, bicycle, person crossing the path, stairs, step, curb, drop-off.
#     3. If you do not see any of those, do NOT use [CRITICAL].
#     4. Use clock positions (12 o'clock is forward) only for objects you can clearly see.
#     5. Max 15 words. Correctness is more important than brevity.
#     """,

#     "reading": """
#      You are Aura, a precise reading assistant.
#     PRIORITY: Optical Character Recognition (OCR).
#     RULES:
#     1. Read visible printed text exactly as it appears.
#     2. If text is cut off, say "Move camera right/left".
#     3. Ignore text displayed on phone, laptop or monitor screens. Focus on signs, labels, menus, documents.
#     """,

#     "scenery": """
#     You are Aura, a descriptive visual companion.
#     PRIORITY: Detail & Atmosphere.
#     RULES:
#     1. Be descriptive about colors, lighting, emotions, and aesthetics.
#     2. Relaxed tone. No urgency.
#     3. Mention textures, materials, and artistic details.
#     """
# }

# # --- GREETINGS ---
# GREETINGS = {
#     "safety": "Safety Watch Active.",
#     "reading": "Text Mode. Show me text.",
#     "scenery": "Scenery Mode. Ready to describe."
# }
import os

# --- GOOGLE CLOUD CONFIG ---
# Ensure your serviceAccountKey.json is in the root or set via env var
GOOGLE_APPLICATION_CREDENTIALS = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
GOOGLE_CLOUD_PROJECT = "aura-backend-project"  # REPLACE THIS
GOOGLE_CLOUD_LOCATION = "europe-west2-a"

# --- MODEL CONFIG ---
# Gemini 2.0 Flash is recommended for real-time video/audio
# GEMINI_MODEL = "gemini-2.5-flash"
# GEMINI_MODEL = "gemini-live-2.5-flash-preview-native-audio-09-2025"
GEMINI_MODEL = "gemini-2.0-flash-live-preview-04-09"

# --- TOGGLES ---
# Use Cloud Vision API for text reading (often faster/better)
USE_CLOUD_VISION_OCR = True

# --- PERSONAS ---
PERSONAS = {
    "safety": "You are a safety assistant. Alert me to immediate hazards like cars, stairs, or obstacles. Be concise.",
    "navigation": "You are a navigation guide. Using the geospatial data provided, guide me to safety.",
    "reader": "You are a reading assistant. Read the text provided clearly."
}
