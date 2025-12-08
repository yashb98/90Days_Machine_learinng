from google import genai
from config import GEMINI_API_KEY


def get_gemini_client():
    if not GEMINI_API_KEY:
        print("⚠️ WARNING: GEMINI_API_KEY is missing.")
        return None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={
                              "api_version": "v1alpha"})
        print("✅ Gemini Client Initialized")
        return client
    except Exception as e:
        print(f"❌ Gemini Client Init Error: {e}")
        return None
