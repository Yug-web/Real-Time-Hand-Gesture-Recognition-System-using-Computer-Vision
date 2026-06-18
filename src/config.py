"""
config.py
----------
Central configuration for GestureAI System Controller.
All tunable parameters live here.
"""

# ── Camera ──────────────────────────────────────────────────────────
CAM_ID            = 0        # Webcam device index (0 = default)
CAM_WIDTH         = 960
CAM_HEIGHT        = 540

# ── MediaPipe ────────────────────────────────────────────────────────
MAX_HANDS         = 1
DETECTION_CONF    = 0.72
TRACKING_CONF     = 0.72

# ── ML Model ─────────────────────────────────────────────────────────
MODEL_PATH        = "models/gesture_model.pkl"
LABEL_PATH        = "models/gesture_labels.pkl"
TRAINING_DATA_DIR = "dataset"
ML_CONFIDENCE     = 0.60     # Minimum ML probability to accept prediction

# ── Gesture Engine ───────────────────────────────────────────────────
SMOOTH_FRAMES      = 7       # Temporal smoothing window (frames)
STABILITY_FRAMES   = 3       # Frames gesture must be same before confirming
PINCH_THRESHOLD    = 42      # Pixel distance for pinch detection
PINCH_SENSITIVITY  = 1.0     # Multiplier (>1 = easier to pinch)

# ── Mouse ─────────────────────────────────────────────────────────────
MOUSE_SMOOTHING    = 7       # EMA smoothing factor (higher = smoother, more lag)
MOUSE_MARGIN       = 100     # px margin inside frame for mouse mapping

# ── Action Cooldowns ──────────────────────────────────────────────────
ACTION_COOLDOWN    = 0.6     # seconds between repeated discrete actions
CLICK_COOLDOWN     = 0.45    # seconds between clicks
MEDIA_COOLDOWN     = 0.8

# ── Volume / Brightness ───────────────────────────────────────────────
CONT_HAND_MIN      = 30      # px — minimum spread for continuous control
CONT_HAND_MAX      = 220     # px — maximum spread

# ── UI ────────────────────────────────────────────────────────────────
WINDOW_TITLE       = "GestureAI — System Controller  |  Press Q to quit"
SHOW_FPS           = True
SHOW_LANDMARKS     = True
