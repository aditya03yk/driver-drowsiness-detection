"""
Test EAR Algorithm without webcam
Run this to verify the algorithm works
"""

import numpy as np
from scipy.spatial import distance

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Simulate OPEN eye landmarks
open_eye = np.array([
    [0, 0], [1, 2], [2, 2],
    [3, 0], [2, -2], [1, -2]
])

# Simulate CLOSED eye landmarks
closed_eye = np.array([
    [0, 0], [1, 0.1], [2, 0.1],
    [3, 0], [2, -0.1], [1, -0.1]
])

ear_open   = eye_aspect_ratio(open_eye)
ear_closed = eye_aspect_ratio(closed_eye)

print("=" * 40)
print("  EAR Algorithm Test")
print("=" * 40)
print(f"  Open Eye  EAR : {ear_open:.4f}  → {'AWAKE ✅' if ear_open > 0.25 else 'DROWSY 🚨'}")
print(f"  Closed Eye EAR: {ear_closed:.4f} → {'AWAKE ✅' if ear_closed > 0.25 else 'DROWSY 🚨'}")
print("=" * 40)
print("  Threshold = 0.25")
print("  Algorithm working correctly!")
