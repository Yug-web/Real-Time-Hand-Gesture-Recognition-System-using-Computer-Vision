"""
main.py
--------
GestureAI — Real-Time Hand Gesture System Controller for Windows.
Controls your ENTIRE system (any app, any window) using hand gestures.

Run:
    python main.py

Keys:
    Q / ESC   → Quit
    R         → Reload ML model from disk
    P         → Pause / Resume gesture control
    +/-       → Increase / decrease mouse smoothing
"""

import cv2
import time
import numpy as np
import sys

from hand_tracking      import HandTracker
from gesture_recognition import GestureRecognizer
from action_controller  import ActionController
from feature_extractor  import FeatureExtractor
import config as cfg


# ─────────────────────────────────────────────────────────────────────
# UI Drawing Helpers
# ─────────────────────────────────────────────────────────────────────

CYAN   = (0, 255, 220)
PURPLE = (200, 60, 255)
WHITE  = (255, 255, 255)
GREEN  = (50, 230, 100)
ORANGE = (30, 165, 255)
RED    = (50,  50, 240)
DARK   = (15, 15, 25)
YELLOW = (0, 230, 255)
BLACK  = (0, 0, 0)

GESTURE_COLORS = {
    "Mouse Move":  CYAN,
    "Left Click":  GREEN,
    "Right Click": PURPLE,
    "Scroll":      YELLOW,
    "Volume":      ORANGE,
    "Brightness":  (0, 220, 255),
    "Play/Pause":  (200, 140, 255),
    "Screenshot":  RED,
    "Zoom In":     (100, 255, 100),
    "Zoom Out":    (80,  80,  255),
    "None":        WHITE,
}

GESTURE_ICONS = {
    "Mouse Move":  "CURSOR",
    "Left Click":  "L-CLICK",
    "Right Click": "R-CLICK",
    "Scroll":      "SCROLL",
    "Volume":      "VOLUME",
    "Brightness":  "BRIGHT",
    "Play/Pause":  "MEDIA",
    "Screenshot":  "SNAP",
    "Zoom In":     "ZOOM+",
    "Zoom Out":    "ZOOM-",
    "None":        "---",
}


def alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.55):
    """Draw a semi-transparent filled rectangle."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    overlay = np.full_like(roi, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)


def draw_bar(frame, x, y, w, h, value, max_val, color, label=""):
    """Draw a horizontal progress bar."""
    alpha_rect(frame, x, y, x + w, y + h, (30, 30, 50), alpha=0.7)
    fill = int((value / max(max_val, 1)) * w)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (x + w + 8, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_hud(
    frame, gesture, conf, fps, vol, bright,
    action_text, action_alpha, paused, hand_present, action_count
):
    h, w = frame.shape[:2]

    # ══ Top-left panel ════════════════════════════════════════════════
    panel_h = 155
    alpha_rect(frame, 0, 0, 300, panel_h, DARK, alpha=0.75)
    cv2.rectangle(frame, (0, 0), (300, panel_h), PURPLE, 1, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "GestureAI", (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, "SYSTEM CONTROLLER", (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, PURPLE, 1, cv2.LINE_AA)
    cv2.line(frame, (10, 54), (290, 54), PURPLE, 1, cv2.LINE_AA)

    # Status dot
    dot_col = GREEN if (hand_present and not paused) else (ORANGE if paused else RED)
    cv2.circle(frame, (20, 70), 6, dot_col, -1, cv2.LINE_AA)
    status_str = "PAUSED" if paused else ("TRACKING" if hand_present else "WAITING")
    cv2.putText(frame, status_str, (34, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, dot_col, 1, cv2.LINE_AA)

    # FPS
    fps_col = GREEN if fps > 25 else (ORANGE if fps > 15 else RED)
    cv2.putText(frame, f"FPS: {int(fps):2d}", (180, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_col, 1, cv2.LINE_AA)

    # Gesture label
    g_col  = GESTURE_COLORS.get(gesture, WHITE)
    g_icon = GESTURE_ICONS.get(gesture, "")
    cv2.putText(frame, f"{g_icon}", (12, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, g_col, 1, cv2.LINE_AA)
    cv2.putText(frame, gesture, (90, 105),
                cv2.FONT_HERSHEY_DUPLEX, 0.68, g_col, 1, cv2.LINE_AA)

    # Confidence bar
    conf_w = int(conf * 180)
    cv2.rectangle(frame, (12, 112), (192, 120), (50, 50, 80), -1)
    cv2.rectangle(frame, (12, 112), (12 + conf_w, 120), g_col, -1)
    cv2.rectangle(frame, (12, 112), (192, 120), g_col, 1)
    cv2.putText(frame, f"{int(conf*100)}%", (198, 121),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, g_col, 1, cv2.LINE_AA)

    # Action count
    cv2.putText(frame, f"Actions: {action_count}", (12, 142),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # ══ Right panel: Volume + Brightness ══════════════════════════════
    rp_x = w - 200
    alpha_rect(frame, rp_x, 0, w, 130, DARK, alpha=0.75)
    cv2.rectangle(frame, (rp_x, 0), (w, 130), PURPLE, 1, cv2.LINE_AA)

    cv2.putText(frame, "VOLUME", (rp_x + 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)
    draw_bar(frame, rp_x + 10, 30, 140, 16, vol, 100, ORANGE, f"{int(vol)}%")

    cv2.putText(frame, "BRIGHTNESS", (rp_x + 10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
    draw_bar(frame, rp_x + 10, 78, 140, 16, bright, 100, YELLOW, f"{int(bright)}%")

    cv2.putText(frame, "P = Pause  Q = Quit", (rp_x + 10, 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 180), 1, cv2.LINE_AA)

    # ══ Action flash (centre-bottom) ══════════════════════════════════
    if action_text and action_alpha > 0:
        alpha_val = min(action_alpha, 1.0)
        txt_size  = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
        tx = (w - txt_size[0]) // 2
        ty = h - 50
        # Shadow
        cv2.putText(frame, action_text, (tx + 2, ty + 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, BLACK, 3, cv2.LINE_AA)
        col = tuple(int(c * alpha_val) for c in CYAN)
        cv2.putText(frame, action_text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2, cv2.LINE_AA)

    # ══ PAUSED overlay ════════════════════════════════════════════════
    if paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        cv2.putText(frame, "PAUSED  — Press P to resume",
                    (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, ORANGE, 2, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n=======================================================")
    print("   GestureAI - Real-Time System Controller")
    print("   Controls mouse, volume, brightness, media & more")
    print("=======================================================\n")
    print("  Q / ESC = Quit    P = Pause    R = Reload ML model\n")

    # ── Init components ──────────────────────────────────────────────
    tracker    = HandTracker(
        max_hands=cfg.MAX_HANDS,
        detection_con=cfg.DETECTION_CONF,
        track_con=cfg.TRACKING_CONF,
        smooth_factor=0.65,
    )
    recognizer = GestureRecognizer()
    extractor  = FeatureExtractor()

    cap = cv2.VideoCapture(cfg.CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAM_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check CAM_ID in config.py")
        sys.exit(1)

    ret, first = cap.read()
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {fw}×{fh}")

    controller = ActionController(frame_w=fw, frame_h=fh)
    print(f"  Screen: {controller._sw}×{controller._sh}")

    # ── State ─────────────────────────────────────────────────────────
    paused        = False
    fps           = 0.0
    fps_history   = []
    prev_time     = time.time()
    action_text   = ""
    action_alpha  = 0.0
    ACTION_FADE   = 1.8
    fail_count    = 0
    MAX_FAILS     = 30

    cv2.namedWindow(cfg.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.WINDOW_TITLE, fw, fh)

    # ── Main loop ─────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count >= MAX_FAILS:
                print("[ERROR] Camera unresponsive. Exiting.")
                break
            time.sleep(0.05)
            continue
        fail_count = 0

        frame = cv2.flip(frame, 1)   # Mirror for natural feel

        # ── Inference ─────────────────────────────────────────────────
        tracker.process(frame)
        hand_present = tracker.hand_detected()

        gesture, conf, extra = "None", 0.0, None

        if hand_present and cfg.SHOW_LANDMARKS:
            tracker.draw_landmarks(frame)

        if hand_present:
            lm_list = tracker.get_landmarks(frame)
            if lm_list:
                gesture, conf, extra = recognizer.recognize(lm_list)

                # ── Execute action ──────────────────────────────────
                if not paused and gesture != "None" and extra is not None:
                    result = controller.handle(gesture, extra)
                    if result:
                        action_text  = result
                        action_alpha = 1.0

        # ── Action flash fade ─────────────────────────────────────────
        now = time.time()
        dt  = now - prev_time
        prev_time = now
        if action_alpha > 0:
            action_alpha = max(0.0, action_alpha - dt / ACTION_FADE)

        # ── FPS ───────────────────────────────────────────────────────
        fps_history.append(1.0 / max(dt, 1e-6))
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps = sum(fps_history) / len(fps_history)

        # ── Draw HUD ──────────────────────────────────────────────────
        frame = draw_hud(
            frame, gesture, conf, fps,
            controller.volume_pct,
            controller.brightness_pct,
            action_text, action_alpha,
            paused, hand_present,
            controller.action_count,
        )

        cv2.imshow(cfg.WINDOW_TITLE, frame)

        # ── Key handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or ESC
            break
        elif key in (ord("p"), ord("P")):
            paused = not paused
            action_text  = "PAUSED" if paused else "RESUMED"
            action_alpha = 1.0
        elif key in (ord("r"), ord("R")):
            recognizer.reload_model()
            action_text  = "Model Reloaded"
            action_alpha = 1.0

    cap.release()
    cv2.destroyAllWindows()
    print("\n[GestureAI] Stopped. Goodbye!")


if __name__ == "__main__":
    main()
