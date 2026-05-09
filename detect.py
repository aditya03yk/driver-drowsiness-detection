"""
Driver Drowsiness Detection System
====================================
USN   : 2102408003
Tech  : Python, OpenCV, MediaPipe, EAR Algorithm, Pygame
"""

import cv2
import numpy as np
from scipy.spatial import distance
import pygame
import time
import urllib.request
import os

# MediaPipe new Tasks API (0.10+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# ─── Constants ────────────────────────────────────────────────
EAR_THRESHOLD     = 0.25
EAR_CONSEC_FRAMES = 20

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# ─── Eye Aspect Ratio ─────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

# ─── Beep Alert ───────────────────────────────────────────────
def init_sound():
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

def play_beep():
    try:
        t = np.linspace(0, 0.4, int(44100 * 0.4), False)
        wave = (np.sin(880 * 2 * np.pi * t) * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(wave)
        sound.play()
    except Exception:
        pass

# ─── Download model if not present ───────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading face landmark model (~5MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded!")

# ─── Main ─────────────────────────────────────────────────────
def main():
    print("[INFO] Starting Driver Drowsiness Detection...")
    print("[INFO] Press 'Q' to quit.\n")

    ensure_model()
    init_sound()

    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1
    )
    detector = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        return

    frame_counter = 0
    alert_on      = False
    alert_start   = None
    last_beep     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.resize(frame, (640, 480))
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        status = "NO FACE"
        color  = (100, 100, 100)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            ear_l = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear   = (ear_l + ear_r) / 2.0

            # Draw eye landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(lm[idx].x * w)
                y = int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= EAR_CONSEC_FRAMES:
                    status = "DROWSY"
                    color  = (0, 0, 255)
                    if not alert_on:
                        alert_on    = True
                        alert_start = time.time()
                    if time.time() - last_beep > 1.0:
                        play_beep()
                        last_beep = time.time()
                    elapsed = time.time() - alert_start
                    cv2.putText(frame, f"DROWSINESS ALERT! ({elapsed:.1f}s)",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(frame, (0, 0), (639, 479), (0, 0, 255), 5)
                else:
                    status = "CLOSING"
                    color  = (0, 165, 255)
            else:
                frame_counter = 0
                alert_on      = False
                alert_start   = None
                status = "AWAKE"
                color  = (0, 255, 0)

            cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame, f"Frames: {frame_counter}/{EAR_CONSEC_FRAMES}",
                        (500, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Status badge
        cv2.rectangle(frame, (8, 8), (220, 38), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {status}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Driver Drowsiness Detection | USN: 2102408003", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("[INFO] Exited.")

if __name__ == "__main__":
    main()
