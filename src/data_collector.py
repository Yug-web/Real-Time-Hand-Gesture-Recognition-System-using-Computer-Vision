"""
data_collector.py
------------------
Custom gesture training tool.

HOW TO USE:
  python data_collector.py

Controls:
  1 → Mouse Move      6 → Brightness
  2 → Left Click      7 → Play/Pause
  3 → Right Click     8 → Screenshot
  4 → Scroll          9 → Zoom In
  5 → Volume          0 → Zoom Out
  N → None gesture
  S → Save collected samples to disk
  R → Retrain model (runs model_trainer.py)
  Q → Quit

Hold the gesture pose while the key is pressed.
Each press captures one sample. Aim for 200+ samples per gesture.
"""

import cv2
import os
import numpy as np
import time

from hand_tracking     import HandTracker
from feature_extractor import FeatureExtractor
from gesture_recognition import GESTURES
import config as cfg

os.makedirs(cfg.TRAINING_DATA_DIR, exist_ok=True)

# ── Key → label mapping ──────────────────────────────────────────
KEY_MAP = {
    ord("1"): "Mouse Move",
    ord("2"): "Left Click",
    ord("3"): "Right Click",
    ord("4"): "Scroll",
    ord("5"): "Volume",
    ord("6"): "Brightness",
    ord("7"): "Play/Pause",
    ord("8"): "Screenshot",
    ord("9"): "Zoom In",
    ord("0"): "Zoom Out",
    ord("n"): "None",
    ord("N"): "None",
}

# UI Colors
CYAN    = (0, 255, 220)
PURPLE  = (200, 60, 255)
WHITE   = (255, 255, 255)
GREEN   = (50, 230, 100)
ORANGE  = (30, 165, 255)
RED     = (50, 50, 240)
BLACK   = (0, 0, 0)
DARK    = (18, 18, 28)


def draw_ui(frame, samples_per_class, current_label, status_msg):
    h, w = frame.shape[:2]

    # ── Left sidebar ─────────────────────────────────────────────────
    panel = np.zeros((h, 280, 3), dtype=np.uint8)
    panel[:] = (18, 18, 28)

    cv2.putText(panel, "GestureAI Trainer", (12, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, CYAN, 1, cv2.LINE_AA)
    cv2.line(panel, (10, 42), (270, 42), PURPLE, 1)

    key_labels = [
        ("1", "Mouse Move"), ("2", "Left Click"),  ("3", "Right Click"),
        ("4", "Scroll"),     ("5", "Volume"),       ("6", "Brightness"),
        ("7", "Play/Pause"), ("8", "Screenshot"),   ("9", "Zoom In"),
        ("0", "Zoom Out"),   ("N", "None"),
    ]
    y = 68
    for key, label in key_labels:
        count = samples_per_class.get(label, 0)
        bar_w = min(int(count / 2.5), 180)
        bar_col = GREEN if count >= 200 else (ORANGE if count >= 100 else RED)

        # Label
        is_active = current_label == label
        col = CYAN if is_active else WHITE
        cv2.putText(panel, f"[{key}] {label}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
        # Sample count
        cv2.putText(panel, f"{count:3d}", (220, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, bar_col, 1, cv2.LINE_AA)
        # Bar
        cv2.rectangle(panel, (12, y + 3), (12 + bar_w, y + 8), bar_col, -1)
        y += 32

    cv2.line(panel, (10, y + 5), (270, y + 5), PURPLE, 1)
    y += 22
    cv2.putText(panel, "[S] Save   [R] Retrain", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    y += 24
    cv2.putText(panel, "[Q] Quit", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # Status message
    y = h - 40
    cv2.putText(panel, status_msg, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREEN, 1, cv2.LINE_AA)

    # Combine
    combined = np.hstack([panel, frame])

    # ── Top banner ────────────────────────────────────────────────────
    banner = np.zeros((44, combined.shape[1], 3), dtype=np.uint8)
    banner[:] = (18, 18, 28)
    total = sum(samples_per_class.values())
    msg = f"Active: [{current_label}]    Total samples: {total}"
    cv2.putText(banner, msg, (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, CYAN, 1, cv2.LINE_AA)
    cv2.line(banner, (0, 43), (combined.shape[1], 43), PURPLE, 1)

    return np.vstack([banner, combined])


def collect():
    tracker   = HandTracker(max_hands=1, detection_con=0.72, track_con=0.72)
    extractor = FeatureExtractor()

    cap = cv2.VideoCapture(cfg.CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAM_HEIGHT)

    samples_per_class: dict = {g: 0 for g in GESTURES}
    collected_X, collected_y = [], []
    current_label = "—"
    status_msg    = "Press a key to start collecting"

    print("\n[Trainer] Press gesture keys to collect samples. S to save, Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        tracker.process(frame)
        lm_list = tracker.get_landmarks(frame)
        frame   = tracker.draw_landmarks(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        elif key == ord("s") or key == ord("S"):
            if collected_X:
                ts   = int(time.time())
                path = os.path.join(cfg.TRAINING_DATA_DIR, f"user_data_{ts}.npz")
                np.savez(path, features=np.array(collected_X), labels=np.array(collected_y))
                status_msg = f"Saved {len(collected_X)} samples → {path}"
                print(f"[Trainer] {status_msg}")
            else:
                status_msg = "Nothing to save yet!"

        elif key == ord("r") or key == ord("R"):
            status_msg = "Retraining model ..."
            cv2.waitKey(50)
            import subprocess, sys
            subprocess.run([sys.executable, "model_trainer.py"])
            status_msg = "Model retrained! Reload main.py."

        elif key in KEY_MAP:
            label = KEY_MAP[key]
            if lm_list:
                feat = extractor.extract(lm_list)
                if feat is not None:
                    collected_X.append(feat)
                    collected_y.append(label)
                    samples_per_class[label] += 1
                    current_label = label
                    status_msg = f"Captured 1 sample for '{label}'"
                else:
                    status_msg = "Feature extraction failed"
            else:
                status_msg = "No hand detected!"

        ui = draw_ui(frame, samples_per_class, current_label, status_msg)
        cv2.imshow("GestureAI — Data Collector  |  Q to quit", ui)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect()
