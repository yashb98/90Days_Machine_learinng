import cv2
import numpy as np
import google.generativeai as genai
import os
import PIL.Image
import time
from dotenv import load_dotenv
from audio_engine import AudioWorker

# --- 1. SETUP ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    STATUS_STATE = "CONFIG_ERROR"
else:
    genai.configure(api_key=API_KEY)
    STATUS_STATE = "IDLE"

# --- 2. RESILIENT AI FUNCTION (With Backoff) ---


def generate_gemini_response(image_path, max_retries=3):
    """
    Tries to get a response. If it fails, waits and tries again (Backoff).
    """
    base_delay = 1  # seconds

    # Try multiple times before giving up
    for attempt in range(max_retries):
        try:
            # Use the model that worked for you (check check_models.py if unsure)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')

            img = PIL.Image.open(image_path)

            # Set timeout to prevent hanging forever
            response = model.generate_content(
                ["Describe exactly what is in this image in one sentence.", img],
                request_options={"timeout": 10}
            )
            return response.text, "SUCCESS"

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))  # Wait 1s, 2s, 4s...
            else:
                return f"Connection Failed. System Offline.", "ERROR"

# --- 3. UI HELPER ---


def draw_status(frame, state):
    """Draws a status indicator on the screen."""
    height, width, _ = frame.shape

    # Defaults (Green/Ready)
    color = (0, 255, 0)
    text = "ONLINE"

    if state == "PROCESSING":
        color = (0, 255, 255)  # Yellow
        text = "THINKING..."
    elif state == "ERROR":
        color = (0, 0, 255)  # Red
        text = "CONNECTION LOST"
    elif state == "RETRYING":
        color = (0, 165, 255)  # Orange
        text = "RETRYING..."

    # Draw Circle Indicator (Top Right)
    cv2.circle(frame, (width - 40, 40), 15, color, -1)

    # Draw Text Label
    cv2.putText(frame, text, (width - 200, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# --- 4. VISION HELPER ---


def has_scene_changed(prev, curr, threshold=40):
    if prev is None:
        return False
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = np.mean(cv2.absdiff(prev_g, curr_g))
    return diff > threshold

# --- 5. MAIN LOOP ---


def main():
    cap = cv2.VideoCapture(0)
    bot_voice = AudioWorker()
    prev_frame = None

    # Current System State
    current_state = "IDLE"

    print("\nVISION AGENT RUNNING (Resilient Mode)")
    print("---------------------------------------")
    print("🟢 GREEN  = Ready")
    print("🟡 YELLOW = Thinking")
    print("🔴 RED    = Error / Offline")
    print("---------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # A. INTERRUPTION LOGIC
        if has_scene_changed(prev_frame, frame):
            bot_voice.stop()
            # If we were waiting for AI, and scene changed, cancel state
            if current_state == "PROCESSING":
                current_state = "IDLE"

        # B. DRAW UI
        draw_status(frame, current_state)

        prev_frame = frame.copy()
        cv2.imshow("Agent View", frame)

        # C. INPUT HANDLING
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and current_state != "PROCESSING":
            print("Requesting AI...")
            current_state = "PROCESSING"
            draw_status(frame, current_state)  # Force update UI immediately
            cv2.imshow("Agent View", frame)
            cv2.waitKey(1)  # process UI event

            # 1. Save Frame
            cv2.imwrite("current_view.jpg", frame)

            # 2. Get Real AI Response (Blocking Call)
            response_text, status = generate_gemini_response(
                "current_view.jpg")

            if status == "ERROR":
                current_state = "ERROR"
                bot_voice.speak("I cannot connect to the brain.")
            else:
                current_state = "IDLE"
                print(f"AI Says: {response_text}")
                bot_voice.speak(response_text)

        # Reset Error state manually if needed
        if key == ord('r'):
            current_state = "IDLE"

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
