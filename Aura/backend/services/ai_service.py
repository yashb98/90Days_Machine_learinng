from google import genai
from google.genai import types
import config
import traceback
import os


class AIService:
    def __init__(self):
        self.client = None
        try:
            print(f"🌍 Connecting to Vertex AI (us-central1)...")
            self.client = genai.Client(
                vertexai=True,
                project=config.GOOGLE_CLOUD_PROJECT,
                location='us-central1',
                http_options=types.HttpOptions(api_version='v1beta1')
            )
            print(f"✅ Vertex AI Service Initialized")
        except Exception as e:
            print(f"❌ Vertex AI Init Error: {e}")
            traceback.print_exc()

    def connect(self):
        """Returns the async context manager for a Live Session."""
        if not self.client:
            raise RuntimeError("Vertex AI Client is not initialized.")

        # 2. Configure the Live Session
        live_config = types.LiveConnectConfig(
            # 👇 CHANGE THIS BACK TO AUDIO
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Puck"  # Voices: Puck, Charon, Kore, Fenrir, Aoede
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(
                    text="You are Aura, an assistive AI. Keep responses short. Warn of danger immediately."
                )]
            )
        )

        return self.client.aio.live.connect(
            model=config.GEMINI_MODEL,
            config=live_config
        )

    async def send_setup_prompt(self, session, mode):
        """Switches the persona based on the selected mode."""
        personas = {
            "safety": "You are a Safety Assistant. Your ONLY job is to identify obstacles, stairs, cars, or hazards. If safe, say 'Path clear'. Keep it under 10 words.",
            "reading": "You are a Reading Assistant. Read any text visible in the frame exactly as it appears. Do not summarize.",
            "scenery": "You are a Describer. Describe the scene, colors, and objects in front of you in detail.",
            "navigation": "You are a Guide. Mention key landmarks and direction."
        }

        instruction = personas.get(mode, personas["safety"])
        print(f"🔄 Sending New Instruction: {instruction}")

        # Send the instruction as text
        await self.send_text(session, f"SYSTEM UPDATE: {instruction}")

    async def send_image_frame(self, session, image_bytes):
        try:
            # 👇 CHANGE 2: REMOVED THE SQUARE BRACKETS []
            # The SDK expects a single Part object, not a list
            await session.send(
                input={"data": image_bytes, "mime_type": "image/jpeg"},
                end_of_turn=False
            )
        except Exception as e:
            print(f"Error sending frame: {e}")

    async def send_text(self, session, text):
        try:
            await session.send(input=text, end_of_turn=True)
        except Exception as e:
            print(f"Error sending text: {e}")
