# 🚗 Driver Drowsiness Detection System

**USN:** 2102408003  


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
