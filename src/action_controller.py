"""
action_controller.py
---------------------
Maps confirmed gestures to REAL Windows system actions:
  - System mouse movement (moves the actual cursor everywhere)
  - Left / right click (works in any app, browser, game)
  - Scroll (works in any scrollable window including YouTube)
  - System volume (Windows Core Audio via pycaw)
  - System brightness (screen-brightness-control)
  - Screenshot (saves to Desktop)
  - Media play/pause / next / previous (global media keys)
  - Zoom in / out (Ctrl+/Ctrl-)
"""

import time
import math
import numpy as np
import pyautogui
import pyautogui as pg
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from typing import Dict, Any, Optional, Tuple
import config as cfg

# Silence PyAutoGUI fail-safe (prevents crash at screen corners)
pyautogui.FAILSAFE = False

# ─── Volume (pycaw) ────────────────────────────────────────────────
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _device    = AudioUtilities.GetSpeakers()
    # New pycaw API (>=20230623): AudioDevice has .EndpointVolume directly
    if hasattr(_device, 'EndpointVolume'):
        _vol_ctrl  = _device.EndpointVolume
    else:
        # Older pycaw fallback
        from comtypes import CLSCTX_ALL
        _iface    = _device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _vol_ctrl  = cast(_iface, POINTER(IAudioEndpointVolume))
    _vol_range = _vol_ctrl.GetVolumeRange()   # (min_dB, max_dB, step_dB)
    _HAS_AUDIO = True
    print(f"[ActionCtrl] System volume control: OK  (range {_vol_range[0]:.1f} to {_vol_range[1]:.1f} dB)")
except Exception as e:
    print(f"[ActionCtrl] Audio init failed: {e}  -- Volume gesture disabled.")
    _vol_ctrl  = None
    _vol_range = (-65.25, 0.0, 0.5)
    _HAS_AUDIO = False

# ─── Brightness (screen-brightness-control) ────────────────────────
try:
    import screen_brightness_control as sbc
    _HAS_BRIGHTNESS = True
except Exception as e:
    print(f"[ActionCtrl] Brightness init failed: {e}  -- Brightness gesture disabled.")
    _HAS_BRIGHTNESS = False


class ActionController:
    """
    Receives a confirmed gesture + extra landmark data and executes the
    corresponding Windows system action.

    Parameters
    ----------
    frame_w, frame_h : int   Camera frame dimensions (pixels).
    """

    def __init__(self, frame_w: int, frame_h: int):
        self._fw = frame_w
        self._fh = frame_h
        self._sw, self._sh = pyautogui.size()   # screen resolution

        # EMA mouse smoothing state
        self._mouse_x: float = self._sw / 2
        self._mouse_y: float = self._sh / 2

        # Cooldowns
        self._t_last_click  = 0.0
        self._t_last_action = 0.0

        # State
        self._action_log: str = ""
        self._action_time: float = 0.0

        # Current volume / brightness (0–100)
        self._volume     = self._read_current_volume()
        self._brightness = self._read_current_brightness()

        # Action counter
        self.action_count = 0

    # ─────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────

    def handle(self, gesture: str, extra: Dict[str, Any]) -> Optional[str]:
        """
        Execute the action for `gesture`.
        Returns a short human-readable description of the action taken,
        or None if no action was triggered (e.g. cooldown).
        """
        if gesture == "None":
            return None

        now = time.time()

        if gesture == "Mouse Move":
            return self._mouse_move(extra)

        if gesture == "Left Click":
            return self._left_click(now, extra)

        if gesture == "Right Click":
            return self._right_click(now)

        if gesture == "Scroll":
            return self._scroll(now, extra)

        if gesture == "Volume":
            return self._volume_control(extra)

        if gesture == "Brightness":
            return self._brightness_control(extra)

        if gesture == "Play/Pause":
            return self._play_pause(now)

        if gesture == "Screenshot":
            return self._screenshot(now)

        if gesture == "Zoom In":
            return self._zoom(now, direction=1)

        if gesture == "Zoom Out":
            return self._zoom(now, direction=-1)

        return None

    @property
    def volume_pct(self) -> int:
        return int(self._volume)

    @property
    def brightness_pct(self) -> int:
        return int(self._brightness)

    # ─────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────

    def _mouse_move(self, extra: Dict) -> str:
        ix, iy = extra["index_tip"]
        margin = cfg.MOUSE_MARGIN

        # Map from frame coords → screen coords
        tx = np.interp(ix, [margin, self._fw - margin], [0, self._sw])
        ty = np.interp(iy, [margin, self._fh - margin], [0, self._sh])

        # Mirror X (camera is mirrored for display)
        tx = self._sw - tx

        # EMA smoothing
        alpha = 1.0 / cfg.MOUSE_SMOOTHING
        self._mouse_x = alpha * tx + (1 - alpha) * self._mouse_x
        self._mouse_y = alpha * ty + (1 - alpha) * self._mouse_y

        pg.moveTo(int(self._mouse_x), int(self._mouse_y))
        return f"Mouse Move ({int(self._mouse_x)}, {int(self._mouse_y)})"

    def _left_click(self, now: float, extra: Dict) -> Optional[str]:
        if now - self._t_last_click < cfg.CLICK_COOLDOWN:
            return None
        self._t_last_click = now
        pg.click()
        self.action_count += 1
        return "Left Click"

    def _right_click(self, now: float) -> Optional[str]:
        if now - self._t_last_click < cfg.CLICK_COOLDOWN:
            return None
        self._t_last_click = now
        pg.rightClick()
        self.action_count += 1
        return "Right Click"

    def _scroll(self, now: float, extra: Dict) -> str:
        iy = extra["index_tip"][1]
        # Top half of frame → scroll up, bottom half → scroll down
        direction = 1 if iy < self._fh * 0.4 else -1
        pg.scroll(direction * 3)
        return "Scroll Up" if direction > 0 else "Scroll Down"

    def _volume_control(self, extra: Dict) -> str:
        d = extra["d_thumb_index"]
        # Map distance to volume 0-100
        vol = float(np.interp(d, [cfg.CONT_HAND_MIN, cfg.CONT_HAND_MAX], [0, 100]))
        self._volume = vol
        if _HAS_AUDIO and _vol_ctrl:
            db = float(np.interp(vol, [0, 100], [_vol_range[0], _vol_range[1]]))
            try:
                _vol_ctrl.SetMasterVolumeLevel(db, None)
            except Exception:
                pass
        return f"Volume {int(vol)}%"

    def _brightness_control(self, extra: Dict) -> str:
        d = extra["d_thumb_middle"]
        bright = float(np.interp(d, [cfg.CONT_HAND_MIN, cfg.CONT_HAND_MAX], [5, 100]))
        self._brightness = bright
        if _HAS_BRIGHTNESS:
            try:
                sbc.set_brightness(int(bright))
            except Exception:
                pass
        return f"Brightness {int(bright)}%"

    def _play_pause(self, now: float) -> Optional[str]:
        if now - self._t_last_action < cfg.MEDIA_COOLDOWN:
            return None
        self._t_last_action = now
        pg.press("playpause")
        self.action_count += 1
        return "Play / Pause"

    def _screenshot(self, now: float) -> Optional[str]:
        if now - self._t_last_action < cfg.ACTION_COOLDOWN * 2:
            return None
        self._t_last_action = now
        import os, datetime
        # Try OneDrive Desktop first, then regular Desktop
        for candidate in [
            os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"),
            os.path.join(os.path.expanduser("~"), "Desktop"),
            os.path.expanduser("~"),
        ]:
            if os.path.isdir(candidate):
                desktop = candidate
                break
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(desktop, f"gesture_screenshot_{ts}.png")
        pg.screenshot(path)
        self.action_count += 1
        return f"Screenshot saved"

    def _zoom(self, now: float, direction: int) -> Optional[str]:
        if now - self._t_last_action < cfg.ACTION_COOLDOWN:
            return None
        self._t_last_action = now
        if direction > 0:
            pg.hotkey("ctrl", "equal")
            self.action_count += 1
            return "Zoom In"
        else:
            pg.hotkey("ctrl", "minus")
            self.action_count += 1
            return "Zoom Out"

    # ─────────────────────────────────────────────────────────────────
    # System reads
    # ─────────────────────────────────────────────────────────────────

    def _read_current_volume(self) -> float:
        if _HAS_AUDIO and _vol_ctrl:
            try:
                db = _vol_ctrl.GetMasterVolumeLevel()
                return float(np.interp(db, [_vol_range[0], _vol_range[1]], [0, 100]))
            except Exception:
                pass
        return 50.0

    def _read_current_brightness(self) -> float:
        if _HAS_BRIGHTNESS:
            try:
                v = sbc.get_brightness()
                if isinstance(v, list):
                    v = v[0]
                return float(v)
            except Exception:
                pass
        return 80.0
