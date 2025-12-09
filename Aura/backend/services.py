import cv2
import numpy as np
import google.generativeai as genai
import asyncio
import io
import PIL.Image
from config import GEMINI_API_KEY

# Initialize Gemini once
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def has_scene_changed(prev_frame, curr_frame, threshold=40):
    """Checks if the visual scene has shifted significantly."""
    if prev_frame is None:
        return False
    try:
        prev_g = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_g = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_g, curr_g)
        return np.mean(diff) > threshold
    except Exception:
        return True


async def generate_gemini_response(image_bytes, prompt_text="Describe this scene."):
    """Sends image to Gemini via REST API."""
    try:
        # Load image from bytes
        image = PIL.Image.open(io.BytesIO(image_bytes))

        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        full_prompt = [
            "You are Aura, a navigation assistant for the blind.",
            "Describe obstacles, directions, and safety hazards in this image concisely.",
            f"User Command: {prompt_text}",
            image
        ]

        # Run blocking I/O in a thread
        response = await asyncio.to_thread(model.generate_content, full_prompt)
        return response.text
    except Exception as e:
        print(f"❌ AI Error: {e}")
        return "I am having trouble seeing."
