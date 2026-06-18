"""
feature_extractor.py
---------------------
Converts raw MediaPipe landmark list → a fixed-length, scale-invariant,
translation-invariant feature vector used by the ML classifier.

Feature breakdown (total = 67 values):
  [0 :42]  — Normalised x,y of 21 landmarks (relative to wrist, scaled by palm size)
  [42:47]  — Finger-up binary flags (thumb, index, middle, ring, pinky)
  [47:57]  — 10 inter-fingertip distances (all pairs of 5 tips)
  [57:62]  — Fingertip-to-wrist distances (5 values)
  [62:67]  — PIP bend angles for each finger (5 values, radians)
"""

import numpy as np
import math
from typing import List, Tuple, Optional


class FeatureExtractor:
    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [2, 6, 10, 14, 18]   # PIP joints (2 below tip)
    MCP_IDS = [1, 5,  9, 13, 17]

    # Finger joint chains used for angle calculation
    FINGER_CHAINS = [
        [1, 2, 3, 4],      # thumb
        [5, 6, 7, 8],      # index
        [9, 10, 11, 12],   # middle
        [13, 14, 15, 16],  # ring
        [17, 18, 19, 20],  # pinky
    ]

    FEATURE_DIM = 67   # 42 + 5 + 10 + 5 + 5

    def extract(
        self, lm_list: List[Tuple[int, int, int]]
    ) -> Optional[np.ndarray]:
        """
        Parameters
        ----------
        lm_list : [(id, px_x, px_y), ...] — raw pixel landmarks from HandTracker.

        Returns
        -------
        np.ndarray of shape (67,), or None if landmarks are invalid.
        """
        if len(lm_list) < 21:
            return None

        # Convert to float array shape (21, 2)
        pts = np.array([[lm[1], lm[2]] for lm in lm_list], dtype=float)

        # ── Normalise: translate wrist to origin ──────────────────────
        wrist = pts[0].copy()
        pts -= wrist

        # ── Normalise: scale by wrist → middle-MCP distance ──────────
        ref = np.linalg.norm(pts[9])
        if ref < 1e-6:
            return None
        pts /= ref

        # ── Feature block 1: flattened normalised positions (42) ──────
        flat = pts.flatten()           # shape (42,)

        # ── Feature block 2: finger-up binary flags (5) ───────────────
        finger_flags = self._fingers_up(lm_list)

        # ── Feature block 3: inter-fingertip distances (10) ──────────
        tips = pts[self.TIP_IDS]       # shape (5, 2)
        tip_dists = []
        for i in range(5):
            for j in range(i + 1, 5):
                tip_dists.append(np.linalg.norm(tips[i] - tips[j]))
        tip_dists = np.array(tip_dists)    # shape (10,)

        # ── Feature block 4: fingertip-to-wrist distances (5) ────────
        wrist_dists = np.array([np.linalg.norm(pts[t]) for t in self.TIP_IDS])

        # ── Feature block 5: finger bend angles (5) ──────────────────
        angles = self._finger_angles(pts)

        features = np.concatenate([flat, finger_flags, tip_dists, wrist_dists, angles])
        assert features.shape == (self.FEATURE_DIM,), f"Feature dim mismatch: {features.shape}"
        return features.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _fingers_up(self, lm_list: List[Tuple[int, int, int]]) -> np.ndarray:
        """Binary vector: 1 = extended, 0 = curled."""
        flags = []
        # Thumb: compare x (left-right) — works for right hand
        flags.append(1.0 if lm_list[4][1] < lm_list[3][1] else 0.0)
        # 4 fingers: tip y < pip y → extended
        for i in range(1, 5):
            tip_y = lm_list[self.TIP_IDS[i]][2]
            pip_y = lm_list[self.PIP_IDS[i]][2]
            flags.append(1.0 if tip_y < pip_y else 0.0)
        return np.array(flags)

    def _finger_angles(self, pts: np.ndarray) -> np.ndarray:
        """
        For each finger, compute the bend angle at the PIP joint using
        the dot product between the proximal and middle phalanx vectors.
        """
        angles = []
        for chain in self.FINGER_CHAINS:
            v1 = pts[chain[1]] - pts[chain[0]]
            v2 = pts[chain[2]] - pts[chain[1]]
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm < 1e-9:
                angles.append(0.0)
            else:
                cos_a = np.dot(v1, v2) / norm
                angles.append(math.acos(float(np.clip(cos_a, -1.0, 1.0))))
        return np.array(angles)

    def fingers_up_raw(self, lm_list: List[Tuple[int, int, int]]) -> List[int]:
        """Return integer finger-up list for rule-based logic."""
        return [int(f) for f in self._fingers_up(lm_list)]
