# Real-Time Hand Gesture Recognition System

A high-performance, real-time hand gesture recognition system designed to control the Windows operating system globally. It leverages a webcam feed to track hand landmarks, classify gestures using a hybrid machine learning + geometric rules model, and execute OS commands (mouse movement, clicks, master volume, screen brightness, and media shortcuts) across all applications.

---

## 📁 Project Structure

```
hand-gesture-recognition/
│
├── dataset/             # User-collected gesture dataset (.npz files)
├── models/              # Pre-trained models and landmarker tasks
│   ├── gesture_labels.pkl
│   ├── gesture_model.pkl
│   └── hand_landmarker.task
├── src/                 # Application source code
│   ├── action_controller.py
│   ├── config.py
│   ├── data_collector.py
│   ├── feature_extractor.py
│   ├── gesture_recognition.py
│   ├── hand_tracking.py
│   ├── main.py
│   └── model_trainer.py
├── .gitignore
└── requirements.txt
```

---

## ✨ System Features & Gesture Controls

The system tracks 10 distinct gestures and maps them to universal OS actions:

| Gesture | Action | Hand Sign Representation | Action Details |
| :--- | :--- | :---: | :--- |
| **Mouse Move** | Cursor Movement | ☝️ | Smooth cursor control via Index Finger position (with EMA smoothing). |
| **Left Click** | Left Mouse Button | 🤏 | Pinch Thumb + Index finger. |
| **Right Click** | Right Mouse Button | 🤞 | Pinch Thumb + Middle finger. |
| **Scroll** | Mouse Wheel Scroll | ✌️ | Raised Index + Middle fingers (scrolls based on vertical position). |
| **Volume** | Continuous Volume | 🤙 | Raised Index + spread Thumb (distance adjusts Master Volume: 0-100%). |
| **Brightness** | Continuous Brightness | 🖕 | Raised Middle + spread Thumb (distance adjusts Backlight: 5-100%). |
| **Play/Pause** | Media Toggle | 👍 | Raised Thumbs Up pose (globally toggles media playback). |
| **Screenshot** | Grab Screen | 🖐️ | 4 fingers up, thumb down (saves high-res screenshot to Desktop). |
| **Zoom In** | Ctrl + Plus | ✋ | Open Palm (spread fingers). |
| **Zoom Out** | Ctrl + Minus | ✊ | Closed Fist. |

---

## 🚀 Getting Started

### 📋 Prerequisites

- **OS**: Windows 10 / 11 (required for direct OS API hookups like volume and brightness)
- **Python**: Version 3.10 to 3.13
- **Hardware**: A working webcam

### ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Yug-web/real-time-hand-gesture-recognition.git
   cd real-time-hand-gesture-recognition
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows Command Prompt:
   venv\Scripts\activate
   # On PowerShell:
   .\venv\Scripts\Activate.ps1
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 How to Run

### 1. Start the Controller
To launch the real-time gesture controller, run:
```bash
python src/main.py
```
*Press **Q** inside the camera window to safely terminate the program.*

### 2. Capture Custom Gestures
To record custom hand poses or train new classes under different lighting conditions, run the data collector utility:
```bash
python src/data_collector.py
```
- Press numerical/alphabetical hotkeys corresponding to gestures (e.g., `1` for Mouse Move, `2` for Left Click, etc.) while holding the pose to save samples.
- Press **S** to compile and export the samples to the `dataset/` directory.
- Press **R** to automatically execute the model trainer on the new data.

### 3. Retrain the Machine Learning Model
To compile both the default synthetic templates and your newly recorded training data into a fresh model, run:
```bash
python src/model_trainer.py
```
This evaluates the pipeline with a 5-fold cross-validation score and writes updated models directly into the `models/` directory.

---

## 🛠 Technology Stack

- **MediaPipe Tasks Vision** — Hand Landmarker models compiled with WASM/Native C++ bindings for real-time 21 hand-joint tracking.
- **Scikit-Learn (Random Forest)** — Predicts complex, non-linear hand posture states with confidence probability gating.
- **PyAutoGUI** — Direct low-level keyboard/mouse driver hooks.
- **Pycaw** — Windows Core Audio API bindings.
- **screen-brightness-control** — WMI-based screen backlight controller.
- **OpenCV** — Multi-threaded video stream acquisition and visualization overlay.

---

## 💡 Troubleshooting

- **No Camera Output**: Change `CAM_ID` in `src/config.py` (e.g., set to `1` or `2` if using an external USB webcam).
- **ModuleNotFoundError on import**: Ensure you run the python scripts from the root directory of the project (e.g. `python src/main.py`) rather than from inside the `src` folder.
- **Mouse Clicks not working in Admin windows**: If you want to click inside task manager or installer wizards, run your terminal/CMD as Administrator before starting the script.
