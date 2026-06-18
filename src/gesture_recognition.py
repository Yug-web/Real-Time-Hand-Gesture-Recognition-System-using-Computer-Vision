"""
gesture_recognition.py
-----------------------
Hybrid gesture classifier:
  1. ML model  (RandomForest trained on synthetic + user data)
  2. Rule-based geometry  (fallback / cross-validation)
  3. Temporal stability filter  (N-frame agreement before confirming)
  4. Confidence gating  (reject low-confidence predictions)

Output: (gesture_name, confidence_0_to_1)
"""

import os
import math
import pickle
import numpy as np
from collections import deque, Counter
from typing import List, Tuple, Optional, Dict, Any

from feature_extractor import FeatureExtractor
import config as cfg


# ─────────────────────────────────────────────────────────────────────
# Gesture label constants
# ─────────────────────────────────────────────────────────────────────
GESTURES = [
    "Mouse Move",    # 0
    "Left Click",    # 1
    "Right Click",   # 2
    "Scroll",        # 3
    "Volume",        # 4
    "Brightness",    # 5
    "Play/Pause",    # 6
    "Screenshot",    # 7
    "Zoom In",       # 8
    "Zoom Out",      # 9
    "None",          # 10
]


class GestureRecognizer:
    """
    Hybrid recogniser combining ML probability + rule-based geometry.

    Parameters
    ----------
    stability_frames : int   Consecutive frames with same prediction before confirming.
    ml_confidence    : float Minimum ML probability to use ML path.
    """

    TIP_IDS = [4, 8, 12, 16, 20]

    def __init__(
        self,
        stability_frames: int   = cfg.STABILITY_FRAMES,
        ml_confidence: float    = cfg.ML_CONFIDENCE,
        smooth_frames: int      = cfg.SMOOTH_FRAMES,
    ):
        self._extractor       = FeatureExtractor()
        self._stability_frames = stability_frames
        self._ml_confidence   = ml_confidence
        self._smooth_frames   = smooth_frames

        # Rolling history for temporal smoothing / stability
        self._pred_history: deque = deque(maxlen=stability_frames)
        self._finger_history: deque = deque(maxlen=smooth_frames)

        # Load ML model if available
        self._model  = None
        self._labels = None
        self._load_model()

        # Last confirmed state
        self._confirmed_gesture = "None"
        self._confirmed_conf    = 0.0

    # ─────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────

    def recognize(
        self, lm_list: List[Tuple[int, int, int]]
    ) -> Tuple[str, float, Optional[Dict[str, Any]]]:
        """
        Returns
        -------
        (gesture_name, confidence, extra_data)

        extra_data contains gesture-specific numeric info (e.g. distances).
        """
        if len(lm_list) < 21:
            self._pred_history.clear()
            self._finger_history.clear()
            return "None", 0.0, None

        # ── Extra data (always computed from raw landmarks) ───────────
        extra = self._compute_extra(lm_list)

        # ── ML prediction ─────────────────────────────────────────────
        ml_name, ml_conf = self._ml_predict(lm_list)

        # ── Rule-based prediction ─────────────────────────────────────
        rule_name = self._rule_predict(lm_list)

        # ── Fusion ───────────────────────────────────────────────────
        if self._model is not None and ml_conf >= self._ml_confidence:
            # ML is confident
            if ml_name == rule_name:
                # Both agree → high confidence
                final_name = ml_name
                final_conf = min(1.0, ml_conf * 1.1)
            else:
                # Disagreement: trust rule-based for safety, mild confidence
                final_name = rule_name
                final_conf = 0.65
        else:
            # No model or low ML confidence → use rules
            final_name = rule_name
            final_conf = 0.80 if rule_name != "None" else 0.0

        # ── Temporal stability filter ──────────────────────────────────
        self._pred_history.append(final_name)
        stable_name = self._most_common(self._pred_history)

        # Require N consecutive same predictions to "confirm"
        if len(self._pred_history) >= self._stability_frames:
            if all(p == stable_name for p in list(self._pred_history)[-self._stability_frames:]):
                self._confirmed_gesture = stable_name
                self._confirmed_conf    = final_conf
            # else: keep previous confirmed gesture
        else:
            self._confirmed_gesture = final_name
            self._confirmed_conf    = final_conf

        return self._confirmed_gesture, self._confirmed_conf, extra

    def reload_model(self):
        self._load_model()

    # ─────────────────────────────────────────────────────────────────
    # ML
    # ─────────────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            if os.path.exists(cfg.MODEL_PATH) and os.path.exists(cfg.LABEL_PATH):
                with open(cfg.MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                with open(cfg.LABEL_PATH, "rb") as f:
                    self._labels = pickle.load(f)
                print(f"[GestureAI] ML model loaded from {cfg.MODEL_PATH}")
            else:
                print("[GestureAI] No ML model found — using rule-based mode only.")
        except Exception as e:
            print(f"[GestureAI] Model load error: {e}")
            self._model  = None
            self._labels = None

    def _ml_predict(self, lm_list) -> Tuple[str, float]:
        if self._model is None:
            return "None", 0.0
        feat = self._extractor.extract(lm_list)
        if feat is None:
            return "None", 0.0
        proba = self._model.predict_proba([feat])[0]
        idx   = int(np.argmax(proba))
        label = self._labels[idx] if self._labels else str(idx)
        return label, float(proba[idx])

    # ─────────────────────────────────────────────────────────────────
    # Rule-based geometry engine
    # ─────────────────────────────────────────────────────────────────

    def _rule_predict(self, lm_list: List[Tuple[int, int, int]]) -> str:
        fingers = self._smooth_fingers(lm_list)

        d_ti = self._dist_px(lm_list, 4, 8)   # thumb ↔ index
        d_tm = self._dist_px(lm_list, 4, 12)  # thumb ↔ middle

        thresh = cfg.PINCH_THRESHOLD * cfg.PINCH_SENSITIVITY

        # ── Priority 1: discrete discrete gestures ────────────────────

        # Screenshot: [0,1,1,1,1] — 4 fingers up, thumb down
        if fingers == [0, 1, 1, 1, 1]:
            return "Screenshot"

        # Play/Pause: [1,0,0,0,0] — thumbs up only
        if fingers == [1, 0, 0, 0, 0]:
            return "Play/Pause"

        # Zoom In: open palm [1,1,1,1,1]
        if fingers == [1, 1, 1, 1, 1]:
            return "Zoom In"

        # Zoom Out: fist [0,0,0,0,0]
        if fingers == [0, 0, 0, 0, 0]:
            return "Zoom Out"

        # ── Priority 2: pinch clicks ──────────────────────────────────

        # Left Click: pinch thumb+index (other fingers down)
        if d_ti < thresh and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            return "Left Click"

        # Right Click: pinch thumb+middle
        if d_tm < thresh and fingers[1] == 0 and fingers[3] == 0 and fingers[4] == 0:
            return "Right Click"

        # ── Priority 3: two-finger scroll ─────────────────────────────
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            return "Scroll"

        # ── Priority 4: continuous controls ──────────────────────────

        # Volume: index up only + thumb spreads
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # Differentiate from Mouse Move by thumb being raised
            if fingers[0] == 1:
                return "Volume"
            return "Mouse Move"

        # Brightness: middle up only + thumb
        if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            return "Brightness"

        return "None"

    def _smooth_fingers(self, lm_list) -> List[int]:
        """Temporally smooth the finger-up vector using majority vote."""
        raw = self._extractor.fingers_up_raw(lm_list)
        self._finger_history.append(raw)
        if len(self._finger_history) < 2:
            return raw
        arr = np.array(self._finger_history)
        return [1 if arr[:, i].mean() >= 0.5 else 0 for i in range(5)]

    # ─────────────────────────────────────────────────────────────────
    # Extra data (gesture-specific numeric info for action controller)
    # ─────────────────────────────────────────────────────────────────

    def _compute_extra(self, lm_list) -> Dict[str, Any]:
        d_ti, mid_ti = self._dist_and_mid(lm_list, 4, 8)
        d_tm, mid_tm = self._dist_and_mid(lm_list, 4, 12)
        return {
            "index_tip":    (lm_list[8][1],  lm_list[8][2]),
            "thumb_tip":    (lm_list[4][1],  lm_list[4][2]),
            "middle_tip":   (lm_list[12][1], lm_list[12][2]),
            "wrist":        (lm_list[0][1],  lm_list[0][2]),
            "d_thumb_index": d_ti,
            "d_thumb_middle": d_tm,
            "mid_ti":       mid_ti,
            "mid_tm":       mid_tm,
        }

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _dist_px(lm_list, i, j) -> float:
        dx = lm_list[i][1] - lm_list[j][1]
        dy = lm_list[i][2] - lm_list[j][2]
        return math.hypot(dx, dy)

    @staticmethod
    def _dist_and_mid(lm_list, i, j):
        x1, y1 = lm_list[i][1], lm_list[i][2]
        x2, y2 = lm_list[j][1], lm_list[j][2]
        return math.hypot(x2 - x1, y2 - y1), ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _most_common(seq) -> str:
        if not seq:
            return "None"
        return Counter(seq).most_common(1)[0][0]
