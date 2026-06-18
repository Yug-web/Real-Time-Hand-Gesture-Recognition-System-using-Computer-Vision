"""
model_trainer.py
-----------------
Generates synthetic training data for all 10 gesture classes and trains
a RandomForestClassifier on it.  Also loads any user-collected samples
from the training_data/ directory to blend in.

Run:
    python model_trainer.py

Output:
    models/gesture_model.pkl
    models/gesture_labels.pkl
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from feature_extractor import FeatureExtractor
from gesture_recognition import GESTURES
import config as cfg

os.makedirs("models", exist_ok=True)

EXTRACTOR = FeatureExtractor()
N_SAMPLES  = 2500   # synthetic samples per gesture


# ═══════════════════════════════════════════════════════════════════
# Synthetic landmark generator
# ═══════════════════════════════════════════════════════════════════

def _jitter(pts, scale=0.03):
    """Add Gaussian noise (scale relative to normalized coords)."""
    return pts + np.random.randn(*pts.shape) * scale

def _make_hand_template():
    """
    Return a base (21,2) hand in normalised coords (wrist=0, palm=1 unit).
    Based on realistic MediaPipe proportions.
    """
    # x, y (wrist origin, y pointing down = finger extended)
    t = np.array([
        [0.00,  0.00],  # 0 wrist
        [-0.15, -0.30], # 1 thumb CMC
        [-0.28, -0.55], # 2 thumb MCP
        [-0.38, -0.73], # 3 thumb IP
        [-0.47, -0.88], # 4 thumb TIP
        [-0.10, -0.60], # 5 index MCP
        [-0.10, -0.85], # 6 index PIP
        [-0.10, -1.00], # 7 index DIP
        [-0.10, -1.10], # 8 index TIP
        [ 0.00, -0.65], # 9 middle MCP
        [ 0.00, -0.92], # 10 middle PIP
        [ 0.00, -1.08], # 11 middle DIP
        [ 0.00, -1.20], # 12 middle TIP
        [ 0.10, -0.62], # 13 ring MCP
        [ 0.10, -0.87], # 14 ring PIP
        [ 0.10, -1.01], # 15 ring DIP
        [ 0.10, -1.12], # 16 ring TIP
        [ 0.20, -0.55], # 17 pinky MCP
        [ 0.20, -0.76], # 18 pinky PIP
        [ 0.20, -0.89], # 19 pinky DIP
        [ 0.20, -0.97], # 20 pinky TIP
    ], dtype=float)
    return t


def _curl_finger(pts, tip_id, pip_id, dip_id, mcp_id, amount=0.9):
    """
    Curl a finger by moving tip/dip/pip towards palm (increase y value).
    amount: 1.0 = fully curled, 0 = straight.
    """
    pts = pts.copy()
    palm_y = pts[mcp_id, 1] + 0.15
    for idx in [tip_id, dip_id, pip_id]:
        pts[idx, 1] += (palm_y - pts[idx, 1]) * amount
    return pts


def _generate_gesture(label: str, n: int) -> np.ndarray:
    """Generate n synthetic samples for the given gesture label."""
    samples = []
    base = _make_hand_template()

    # Helper: finger tip/dip/pip/mcp ids
    F = {
        "thumb":  (4, 3, 2, 1),
        "index":  (8, 7, 6, 5),
        "middle": (12, 11, 10, 9),
        "ring":   (16, 15, 14, 13),
        "pinky":  (20, 19, 18, 17),
    }

    for _ in range(n):
        pts = base.copy()

        if label == "Mouse Move":
            # index up, rest curled
            pts = _curl_finger(pts, *F["thumb"],  amount=np.random.uniform(0.5, 1.0))
            pts = _curl_finger(pts, *F["middle"], amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["ring"],   amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["pinky"],  amount=np.random.uniform(0.7, 1.0))

        elif label == "Left Click":
            # thumb tip close to index tip (pinch) — all others curled
            pts = _curl_finger(pts, *F["middle"], amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["ring"],   amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["pinky"],  amount=np.random.uniform(0.7, 1.0))
            # Move thumb tip to meet index tip
            mid = (pts[4] + pts[8]) / 2
            pts[4] = mid + np.random.randn(2) * 0.02
            pts[8] = mid + np.random.randn(2) * 0.02

        elif label == "Right Click":
            # thumb tip close to middle tip
            pts = _curl_finger(pts, *F["index"],  amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["ring"],   amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["pinky"],  amount=np.random.uniform(0.7, 1.0))
            mid = (pts[4] + pts[12]) / 2
            pts[4]  = mid + np.random.randn(2) * 0.02
            pts[12] = mid + np.random.randn(2) * 0.02

        elif label == "Scroll":
            # index + middle up, others curled
            pts = _curl_finger(pts, *F["thumb"],  amount=np.random.uniform(0.5, 1.0))
            pts = _curl_finger(pts, *F["ring"],   amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["pinky"],  amount=np.random.uniform(0.7, 1.0))

        elif label == "Volume":
            # index up, thumb spread (not curled), others curled
            pts = _curl_finger(pts, *F["middle"], amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["ring"],   amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["pinky"],  amount=np.random.uniform(0.7, 1.0))
            # Spread thumb slightly (not curled, not fully extended)
            pts[4, 0] -= np.random.uniform(0.1, 0.3)

        elif label == "Brightness":
            # middle up, thumb spread, rest curled
            pts = _curl_finger(pts, *F["index"],  amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["ring"],   amount=np.random.uniform(0.7, 1.0))
            pts = _curl_finger(pts, *F["pinky"],  amount=np.random.uniform(0.7, 1.0))
            pts[4, 0] -= np.random.uniform(0.1, 0.3)

        elif label == "Play/Pause":
            # thumbs up only
            pts = _curl_finger(pts, *F["index"],  amount=np.random.uniform(0.8, 1.0))
            pts = _curl_finger(pts, *F["middle"], amount=np.random.uniform(0.8, 1.0))
            pts = _curl_finger(pts, *F["ring"],   amount=np.random.uniform(0.8, 1.0))
            pts = _curl_finger(pts, *F["pinky"],  amount=np.random.uniform(0.8, 1.0))

        elif label == "Screenshot":
            # 4 fingers up, thumb curled
            pts = _curl_finger(pts, *F["thumb"],  amount=np.random.uniform(0.7, 1.0))

        elif label == "Zoom In":
            pass  # Open palm — all fingers extended (base template)

        elif label == "Zoom Out":
            # Full fist
            pts = _curl_finger(pts, *F["thumb"],  amount=np.random.uniform(0.6, 0.9))
            pts = _curl_finger(pts, *F["index"],  amount=np.random.uniform(0.8, 1.0))
            pts = _curl_finger(pts, *F["middle"], amount=np.random.uniform(0.8, 1.0))
            pts = _curl_finger(pts, *F["ring"],   amount=np.random.uniform(0.8, 1.0))
            pts = _curl_finger(pts, *F["pinky"],  amount=np.random.uniform(0.8, 1.0))

        elif label == "None":
            # Random partial hand pose
            for fname, fids in F.items():
                pts = _curl_finger(pts, *fids, amount=np.random.uniform(0, 1.0))

        # Random global scale, rotation, translation
        scale = np.random.uniform(0.8, 1.3)
        angle = np.random.uniform(-0.3, 0.3)
        rot   = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle),  np.cos(angle)]])
        pts   = (pts * scale) @ rot.T
        pts   += np.random.randn(2) * 0.05   # small global translation

        # Jitter
        pts = _jitter(pts, scale=np.random.uniform(0.01, 0.04))

        # Convert to lm_list format [(id, px_x, px_y)] using pixel-like coords
        # Rescale so wrist→middle-MCP ≈ 200px (realistic)
        pts_px = pts * 200 + np.array([320, 240])
        lm_list = [(i, int(pts_px[i, 0]), int(pts_px[i, 1])) for i in range(21)]

        feat = EXTRACTOR.extract(lm_list)
        if feat is not None:
            samples.append(feat)

    return np.array(samples)


# ═══════════════════════════════════════════════════════════════════
# Load user-collected samples
# ═══════════════════════════════════════════════════════════════════

def load_user_data():
    X_user, y_user = [], []
    td = cfg.TRAINING_DATA_DIR
    if not os.path.exists(td):
        return np.array([]), np.array([])
    for fname in os.listdir(td):
        if not fname.endswith(".npz"):
            continue
        data = np.load(os.path.join(td, fname), allow_pickle=True)
        X_user.extend(data["features"])
        y_user.extend(data["labels"])
        print(f"  Loaded user data: {fname}  ({len(data['features'])} samples)")
    if X_user:
        return np.array(X_user), np.array(y_user)
    return np.array([]), np.array([])


# ═══════════════════════════════════════════════════════════════════
# Main training routine
# ═══════════════════════════════════════════════════════════════════

def train():
    print("\n===========================================")
    print("   GestureAI - Model Trainer")
    print("===========================================\n")

    # ── Generate synthetic data ───────────────────────────────────────
    gesture_classes = [g for g in GESTURES if g != "None"]
    gesture_classes.append("None")

    X_synth, y_synth = [], []
    for label in gesture_classes:
        print(f"  Generating synthetic data for '{label}' ...", end="\r")
        samples = _generate_gesture(label, N_SAMPLES)
        X_synth.append(samples)
        y_synth.extend([label] * len(samples))
        print(f"  ✓ '{label}':  {len(samples)} samples")

    X_synth = np.vstack(X_synth)
    y_synth = np.array(y_synth)

    # ── Load user data ────────────────────────────────────────────────
    X_user, y_user = load_user_data()
    if len(X_user) > 0:
        X_all = np.vstack([X_synth, X_user])
        y_all = np.concatenate([y_synth, y_user])
        print(f"\n  Combined: {len(X_synth)} synthetic + {len(X_user)} user samples")
    else:
        X_all, y_all = X_synth, y_synth
        print(f"\n  Using {len(X_all)} synthetic samples  (no user data found)")

    # ── Train/test split ──────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)

    # ── Build ensemble model ──────────────────────────────────────────
    print("\n  Training RandomForest classifier ...")
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ))
    ])
    rf.fit(X_tr, y_tr)

    # ── Evaluate ──────────────────────────────────────────────────────
    acc = rf.score(X_te, y_te)
    print(f"\n  Test accuracy: {acc*100:.1f}%")

    cv_scores = cross_val_score(rf, X_all, y_all, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  5-fold CV:     {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    print("\n" + classification_report(y_te, rf.predict(X_te), zero_division=0))

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open(cfg.MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)
    with open(cfg.LABEL_PATH, "wb") as f:
        pickle.dump(gesture_classes, f)

    print(f"  ✅ Model saved → {cfg.MODEL_PATH}")
    print(f"  ✅ Labels saved → {cfg.LABEL_PATH}")
    print("\nDone! Run  python main.py  to start the gesture controller.\n")


if __name__ == "__main__":
    train()
