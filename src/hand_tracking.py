"""
hand_tracking.py
-----------------
Hand tracker using MediaPipe Tasks API (mediapipe >= 0.10).
Uses HandLandmarker with EMA smoothing for stable, jitter-free landmarks.
Downloads the hand_landmarker.task model on first run.
"""

import cv2
import urllib.request
import os
import numpy as np
from typing import List, Tuple, Optional

from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode


MODEL_PATH = "models/hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe hand connections (21 landmark pairs)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

TIP_IDS = [4, 8, 12, 16, 20]

# Drawing colours (BGR)
COLOR_CONNECTION = (180, 60, 255)   # purple
COLOR_LANDMARK   = (0, 255, 220)    # cyan
COLOR_TIP        = (255, 255, 255)  # white


def _download_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"[HandTracker] Downloading hand landmarker model ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[HandTracker] Model saved to {MODEL_PATH}")


class HandTracker:
    """
    Real-time hand landmark tracker using MediaPipe Tasks HandLandmarker.

    Parameters
    ----------
    max_hands     : int   Maximum simultaneous hands.
    detection_con : float Detection confidence threshold.
    track_con     : float Tracking confidence threshold.
    smooth_factor : float EMA alpha for position smoothing (0=max smooth, 1=raw).
    """

    def __init__(
        self,
        max_hands: int       = 1,
        detection_con: float = 0.72,
        track_con: float     = 0.72,
        smooth_factor: float = 0.65,
    ):
        _download_model()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=track_con,
        )
        self._detector   = HandLandmarker.create_from_options(options)
        self._alpha      = smooth_factor
        self._ema_lm: Optional[np.ndarray] = None
        self._last_result = None
        self._timestamp  = 0

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run inference on a BGR frame. Call before get_landmarks()."""
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp += 33   # ~30fps timestamps in ms
        self._last_result = self._detector.detect_for_video(mp_image, self._timestamp)
        return frame

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw hand skeleton on frame in place."""
        lm_list = self.get_landmarks(frame, apply_ema=False)
        if not lm_list:
            return frame
        h, w = frame.shape[:2]
        pts = {lm[0]: (lm[1], lm[2]) for lm in lm_list}

        # Connections
        for (a, b) in HAND_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], COLOR_CONNECTION, 2, cv2.LINE_AA)

        # Joints
        for i, (px, py) in pts.items():
            is_tip = i in TIP_IDS
            r = 7 if is_tip else 4
            col = COLOR_TIP if is_tip else COLOR_LANDMARK
            cv2.circle(frame, (px, py), r, col, cv2.FILLED, cv2.LINE_AA)
            if is_tip:
                cv2.circle(frame, (px, py), r + 2, COLOR_LANDMARK, 1, cv2.LINE_AA)
        return frame

    def get_landmarks(
        self, frame: np.ndarray, hand_no: int = 0, apply_ema: bool = True
    ) -> List[Tuple[int, int, int]]:
        """
        Return [(id, px, py), ...] for the detected hand.
        Applies EMA smoothing when apply_ema=True.
        """
        if not self._last_result or not self._last_result.hand_landmarks:
            self._ema_lm = None
            return []
        if hand_no >= len(self._last_result.hand_landmarks):
            return []

        h, w = frame.shape[:2]
        raw = np.array(
            [[lm.x * w, lm.y * h]
             for lm in self._last_result.hand_landmarks[hand_no]],
            dtype=float,
        )

        if apply_ema:
            if self._ema_lm is None or self._ema_lm.shape != raw.shape:
                self._ema_lm = raw.copy()
            else:
                self._ema_lm = self._alpha * raw + (1 - self._alpha) * self._ema_lm
            pts = self._ema_lm
        else:
            pts = raw

        return [(i, int(pts[i][0]), int(pts[i][1])) for i in range(len(pts))]

    def hand_detected(self) -> bool:
        return bool(
            self._last_result and self._last_result.hand_landmarks
        )

    def get_bounding_box(self, lm_list: List[Tuple[int, int, int]], pad: int = 20):
        if not lm_list:
            return None
        xs = [lm[1] for lm in lm_list]
        ys = [lm[2] for lm in lm_list]
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
