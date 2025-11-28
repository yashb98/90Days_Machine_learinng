import cv2
import numpy as np
import google.generativeai as genai
import os
import PIL.Image
from dotenv import load_dotenv
# Ensure audio_engine.py is in the same folder
from audio_engine import AudioWorker

# --- 1. SETUP ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure this matches your .env name

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
else:
    # Configure the Standard V1 SDK
    genai.configure(api_key=API_KEY)
    print("Gemini Configured")

# --- 2. AI FUNCTION ---


def generate_gemini_response(image_path):
    """
    Sends the image to Gemini 1.5 Flash and returns the text description.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # We use PIL to open the image locally (faster than File API upload)
        img = PIL.Image.open(image_path)

        response = model.generate_content(
            ["Describe exactly what is in this image in one sentence.", img])
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# --- 3. VISION HELPER ---


def has_scene_changed(prev, curr, threshold=40):
    """Detects if the camera view has shifted significantly."""
    if prev is None:
        return False
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = np.mean(cv2.absdiff(prev_g, curr_g))
    return diff > threshold

# --- 4. MAIN LOOP ---


def main():
    cap = cv2.VideoCapture(0)

    # Initialize the background audio engine
    bot_voice = AudioWorker()
    prev_frame = None

    print("\n VISION AGENT RUNNING")
    print("---------------------------------------")
    print("PRESS 's' --> Agent looks and speaks")
    print("MOVE CAMERA --> Interrupts the agent")
    print("PRESS 'q' --> Quit")
    print("---------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        # A. INTERRUPTION LOGIC (The "Eyes")
        if has_scene_changed(prev_frame, frame):
            # If the audio is playing, this KILL command stops it instantly
            bot_voice.stop()

            # Visual feedback
            cv2.putText(frame, "INTERRUPT TRIGGERED!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        prev_frame = frame.copy()
        cv2.imshow("Agent View", frame)

        # B. INPUT HANDLING
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("Thinking...")

            # 1. Save Frame
            cv2.imwrite("current_view.jpg", frame)

            # 2. Get Real AI Response
            response_text = generate_gemini_response("current_view.jpg")
            print(f"AI Says: {response_text}")

            # 3. Speak (This runs in background thread)
            bot_voice.speak(response_text)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
