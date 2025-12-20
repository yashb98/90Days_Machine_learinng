import traceback
from google import genai
from google.genai import types
import config


class AIService:
    def __init__(self):
        # Initialize Client based on Config
        # This logic is now hidden from the rest of the app
        self.client = None
        try:
            if config.USE_VERTEX_AI:
                print(
                    f"🌍 Connecting to Vertex AI ({config.GOOGLE_CLOUD_LOCATION})...")
                self.client = genai.Client(
                    vertexai=True,
                    project=config.GOOGLE_CLOUD_PROJECT,
                    location=config.GOOGLE_CLOUD_LOCATION,
                    http_options=types.HttpOptions(api_version='v1beta1')
                )
            else:
                self.client = genai.Client(
                    api_key=config.GEMINI_API_KEY,
                    http_options=types.HttpOptions(api_version='v1beta1')
                )
            print("✅ AI Service Initialized")
        except Exception as e:
            print(f"❌ AI Init Error: {e}")

    def connect(self):
        """Returns the async context manager for a Live Session."""
        if not self.client:
            raise RuntimeError("AI Client is not initialized.")

        # Zero-config handshake to avoid 400 errors
        return self.client.aio.live.connect(
            model=config.GEMINI_MODEL,
            config=None
        )

    async def send_setup_prompt(self, session, mode):
        """Sends the initial persona instructions."""
        system_instruction = config.PERSONAS.get(
            mode, config.PERSONAS["safety"])
        greeting = config.GREETINGS.get(mode, "System Online.")

        prompt = (
            f"SYSTEM_INSTRUCTION: {system_instruction}\n"
            f"TASK: You are a real-time vision assistant. "
            f"Keep responses short and concise. "
            f"Say '{greeting}' to confirm you are ready."
        )

        await session.send(input=prompt, end_of_turn=True)

    async def send_image_frame(self, session, image_bytes):
        """Wraps raw bytes into the specific Pydantic objects SDK needs."""
        image_blob = types.Blob(
            data=image_bytes,
            mime_type="image/jpeg"
        )
        # Wrap in List to avoid "Unsupported input type" error
        await session.send(input=[types.Part(inline_data=image_blob)], end_of_turn=False)

    async def send_text(self, session, text):
        await session.send(input=text, end_of_turn=True)
