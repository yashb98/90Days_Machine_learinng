# This uses the Cloud Vision API. It is specialised for high-fidelity OCR.
# If the user asks to "Read this", we route here instead of the LLM for precision

from google.cloud import vision
from google.oauth2 import service_account
import io


class VisionService:
    def __init__(self):
        # The client automatically uses GOOGLE_APPLICATION_CREDENTIALS
        self.client = vision.ImageAnnotatorClient()
        print("✅ Cloud Vision Service Initialized")

    def detect_text(self, image_bytes):
        """
        Uses Google Cloud Vision API to detect text in an image.
        Returns the full text annotation.
        """
        try:
            image = vision.Image(content=image_bytes)

            # Perform text detection
            response = self.client.text_detection(image=image)
            texts = response.text_annotations

            if texts:
                # The first element is the full text
                return texts[0].description.strip()
            return "No text detected."

        except Exception as e:
            print(f"❌ Vision API Error: {e}")
            return None

    def detect_labels(self, image_bytes):
        """
        Uses Cloud Vision to detect objects/labels (useful for metadata).
        """
        try:
            image = vision.Image(content=image_bytes)
            response = self.client.label_detection(image=image)
            return [label.description for label in response.label_annotations]
        except Exception as e:
            return []
