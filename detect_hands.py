import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

def detect_hands_in_image(image: np.ndarray,
                          min_detection_confidence: float = 0.5,
                          min_tracking_confidence: float = 0.5,
                          max_num_hands: int = 2):
    """
    Detects hands in an image using MediaPipe.

    Returns:
        A list[dict], each dict has:
           {
               'bbox': (x_min, y_min, x_max, y_max),
               'landmarks': [(x0, y0), ..., (x20, y20)]
           }
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_dicts = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    ) as hands:
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract bounding box from landmarks
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(min(x_coords) * w)
                x_max = int(max(x_coords) * w)
                y_min = int(min(y_coords) * h)
                y_max = int(max(y_coords) * h)

                # Store all landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append((int(lm.x * w), int(lm.y * h)))

                results_dicts.append({
                    "bbox": (x_min, y_min, x_max, y_max),
                    "landmarks": landmarks
                })

    return results_dicts


def detect_hands_in_first_frame(video_path: str):
    """
    Opens the video, reads the first frame,
    runs detect_hands_in_image, and returns bounding boxes.
    """
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        print(f"[Error] Could not read first frame from: {video_path}")
        cap.release()
        return []

    hand_info = detect_hands_in_image(frame)
    cap.release()
    return hand_info


if __name__ == "__main__":
    # Simple test
    test_video = "test.mp4"
    hands_info = detect_hands_in_first_frame(test_video)
    print("Detected hand bounding boxes from first frame:")
    for h in hands_info:
        print(h["bbox"])
