from google import genai
from google.genai import types
import config
import traceback

# This Manages the VertexAI connection using the unified genai SDK


class AIService:
    def __init__(self):
        self.client = None
        try:
            print(
                f"🌍 Connecting to Vertex AI ({config.GOOGLE_CLOUD_LOCATION})...")

            # Initialize Client for Vertex AI
            self.client = genai.Client(
                vertexai=True,
                project=config.GOOGLE_CLOUD_PROJECT,
                location=config.GOOGLE_CLOUD_LOCATION,
                http_options=types.HttpOptions(api_version='v1beta1')
            )
            print("✅ Vertex AI Service Initialized")
        except Exception as e:
            print(f"❌ Vertex AI Init Error: {e}")
            traceback.print_exc()

    def connect(self):
        """Returns the async context manager for a Live Session."""
        if not self.client:
            raise RuntimeError("Vertex AI Client is not initialized.")

        # Configuration for the Live API
        # We can add 'tools' here if we want the model to call functions
        live_config = types.LiveConnectConfig(
            # or ["TEXT"] based on your app needs
            response_modalities=["AUDIO"]
        )

        return self.client.aio.live.connect(
            model=config.GEMINI_MODEL,
            config=live_config
        )

    async def send_setup_prompt(self, session, mode):
        instruction = config.PERSONAS.get(mode, config.PERSONAS["safety"])
        prompt = f"SYSTEM: {instruction} Confirm ready."
        await session.send(input=prompt, end_of_turn=True)

    async def send_image_frame(self, session, image_bytes):
        """Sends a video frame to the live session."""
        await session.send(
            input=[types.Part.from_bytes(image_bytes, "image/jpeg")],
            end_of_turn=False
        )

    async def send_text(self, session, text):
        await session.send(input=text, end_of_turn=True)
