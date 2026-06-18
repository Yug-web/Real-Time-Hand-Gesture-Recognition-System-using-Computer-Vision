# Real-Time Hand Gesture Recognition System using Computer Vision

A Python-based desktop application that allows you to control your Windows PC using hand gestures. It uses MediaPipe for hand landmark tracking and a Random Forest classifier to recognize gestures in real-time.

## Features & Controls

| Gesture | Action |
| --- | --- |
| Index finger up | Mouse Movement |
| Pinch (Thumb + Index) | Left Click |
| Pinch (Thumb + Middle) | Right Click |
| Index & Middle up | Scroll Up/Down |
| Index up + spread Thumb | Volume Control |
| Middle up + spread Thumb | Brightness Control |
| Thumbs Up | Play / Pause |
| 4 Fingers up | Screenshot |
| Open Palm | Zoom In |
| Closed Fist | Zoom Out |

## Project Structure

```
hand-gesture-recognition/
├── dataset/
├── models/
│   ├── gesture_labels.pkl
│   ├── gesture_model.pkl
│   └── hand_landmarker.task
├── src/
│   ├── action_controller.py
│   ├── config.py
│   ├── data_collector.py
│   ├── feature_extractor.py
│   ├── gesture_recognition.py
│   ├── hand_tracking.py
│   ├── main.py
│   └── model_trainer.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Yug-web/Real-Time-Hand-Gesture-Recognition-System-using-Computer-Vision.git
   cd Real-Time-Hand-Gesture-Recognition-System-using-Computer-Vision
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Run the controller**:
   ```bash
   python src/main.py
   ```
   *Press 'q' to quit.*

2. **Collect custom data**:
   ```bash
   python src/data_collector.py
   ```
   *Press gesture keys (1-9, 0, n) to save samples, 's' to save, and 'r' to retrain.*

3. **Train the model**:
   ```bash
   python src/model_trainer.py
   ```
