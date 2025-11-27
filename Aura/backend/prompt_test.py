import os
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types
import asyncio

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# --- AURA SYSTEM INSTRUCTION ---
AURA_SYSTEM_INSTRUCTION = """
You are AURA, an emergency safety and navigation guide for the visually impaired. Your primary goal is to provide **immediate, actionable, and concise** audio descriptions based on the provided visual and auditory input.

**PRIORITIES (in order):**
1.  **Imminent Danger/Safety:** Instantly report moving vehicles, open doors, stairs (up or down), sudden drops, or immediate obstacles (people, poles, debris). Use urgent, brief language.
2.  **Reading Text:** If the user holds a sign, label, or large text clearly in view, read it out briefly.
3.  **Environmental Context:** Describe the general environment (e.g., "You are on a busy street corner," "Clear path ahead").
4.  **Format:** Your output will be spoken aloud. DO NOT use preamble, markdown, lists, or conversational filler (e.g., "Certainly," "I see"). Respond in short, direct sentences.
"""


def test_aura_persona(prompt: str):
    """Initializes Gemini and sends a multimodal request with the System Instruction."""
    if not API_KEY:
        print("FATAL: GEMINI_API_KEY is not set in .env file.")
        return

    try:
        # Initialize the client
        client = genai.Client(api_key=API_KEY)

        # We will use the base gemini-2.5-flash for speed test
        model = "gemini-2.5-flash"

        # 1. Create a placeholder image (since we don't have a live camera)
        # We use a simple placeholder image for stability.
        # This base64 is for a 1x1 black pixel image (minimal size).
        image_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
            types.Part.from_text(text=prompt)
        ]

        print(f"Sending prompt to {model}...")

        # Measure latency for initial feedback
        start_time = asyncio.get_event_loop().time()

        # 2. Call the API with the System Instruction
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=AURA_SYSTEM_INSTRUCTION
            )
        )

        end_time = asyncio.get_event_loop().time()
        latency_ms = int((end_time - start_time) * 1000)

        # 3. Print Results
        print("-" * 50)
        print(f"Response Time: {latency_ms} ms")
        print(f"AURA's Response:\n{response.text.strip()}")
        print("-" * 50)

    except Exception as e:
        print(f"API Call Failed: {e}")


if __name__ == "__main__":
    # Note: If running outside of an asyncio loop environment, use a simple sync call:
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass  # Use default asyncio loop if uvloop not installed

    # --- Test Scenarios ---
    # 1. Test Safety Priority
    test_aura_persona(
        "Describe the imminent danger: I see a car moving quickly toward me.")

    # 2. Test Reading Text Priority
    test_aura_persona(
        "Describe the text I am pointing at: The sign says 'Library Hours'.")

    # 3. Test Conciseness/Filler Filter
    test_aura_persona(
        "Can you describe the general area I am in? I am standing near a park bench.")
