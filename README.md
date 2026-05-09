# 🚗 Driver Drowsiness Detection System

**USN:** 2102408003  
**Subject:** Machine Learning (Sem 4, B.Tech CSE)  
**Tech Stack:** Python, OpenCV, dlib, EAR Algorithm, Pygame

---

## 📌 Problem Statement
Drowsy driving causes thousands of accidents every year. This system uses a webcam to monitor the driver's eyes in real-time and triggers an alert if drowsiness is detected using the Eye Aspect Ratio (EAR) algorithm.

---

## 🧠 How It Works

```
Webcam Feed
    │
    ▼
Face Detection (dlib HOG detector)
    │
    ▼
68 Facial Landmark Detection
    │
    ▼
Eye Region Extraction (Left + Right)
    │
    ▼
EAR Calculation
  EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    │
    ├── EAR > 0.25 → AWAKE ✅
    │
    └── EAR < 0.25 for 20 frames → DROWSY 🚨 → ALERT!
```

---

## ⚙️ Setup Instructions

### Step 1 — Install Python libraries
```bash
pip install -r requirements.txt
```

### Step 2 — Download dlib shape predictor model
Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
Extract and place `shape_predictor_68_face_landmarks.dat` in the same folder.

### Step 3 — Run
```bash
python detect.py
```
Press **Q** to quit.

---

## 📁 Project Structure
```
drowsiness_detection/
├── detect.py                          ← Main program
├── requirements.txt                   ← Dependencies
├── shape_predictor_68_face_landmarks.dat  ← dlib model (download separately)
└── README.md                          ← This file
```

---

## 🔬 Algorithm — Eye Aspect Ratio (EAR)

The EAR is calculated using 6 landmark points around each eye:

```
EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```

- When eye is **open** → EAR ≈ 0.3
- When eye is **closed** → EAR ≈ 0.0
- **Threshold:** EAR < 0.25 for 20 consecutive frames → Drowsy

---

## 📊 Results

| Condition | EAR Value | Detection |
|-----------|-----------|-----------|
| Eyes Open | ~0.30     | AWAKE ✅  |
| Eyes Half | ~0.20     | Warning ⚠️ |
| Eyes Closed | ~0.05   | ALERT 🚨  |

---

## 🌍 Real-World Applications
- Highway driver monitoring systems
- Commercial truck/fleet safety
- Embedded in dashcam hardware
- Integration with autonomous vehicles

---

## 🔮 Future Scope
- Add yawning detection
- Mobile app version
- GPS-based auto-stop feature
- Dataset training for custom ML model
