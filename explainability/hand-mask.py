import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def enhance_image_for_hand_detection(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = cv2.resize(img_bgr, (512, 512))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def generate_hand_mask(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.shape[0] == 1:
        image_chw = np.repeat(image_chw, 3, axis=0)
    img = (image_chw * 255).astype(np.uint8) if image_chw.max() <= 1.0 else image_chw.astype(np.uint8)
    img = img.transpose(1,2,0)  # HWC RGB
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgr = enhance_image_for_hand_detection(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w, _ = rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(mask, (x, y), 20, 255, -1)
    return (mask > 0).astype(np.uint8)
