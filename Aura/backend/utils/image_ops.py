import cv2
import numpy as np


def calculate_motion_score(current_bytes, previous_frame_gray):
    """
    Calculates percentage of motion between two frames.
    Returns: (score, new_gray_frame)
    """
    nparr = np.frombuffer(current_bytes, np.uint8)
    current_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if current_frame is None:
        return 0.0, previous_frame_gray

    # Convert to grayscale and blur to remove noise
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame_gray is None:
        return 100.0, gray

    # Calculate difference
    frame_delta = cv2.absdiff(previous_frame_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Calculate score (percentage of changed pixels)
    score = (np.count_nonzero(thresh) / thresh.size) * 100
    return score, gray
