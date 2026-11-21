"""Neuro-Flow webcam perception core (professor-update, hip-free 2D primary).

    Camera frame -> MediaPipe IMAGE landmarks -> pixel coordinates -> arm elevation.
    The hip is diagnostic only; it is NOT required for arm elevation.
    Do not mix normalized x/y, pixel, and MediaPipe world coordinates.
"""
import csv
import json
import math
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark, PoseLandmarksConnections

from geometry import arm_elevation, shoulder_line_tilt, tilt_change

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "pose_landmarker_lite.task"
SETTINGS_PATH = ROOT / "settings.json"
CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

JOINTS = {
    "right": {"hip": PoseLandmark.RIGHT_HIP.value, "shoulder": PoseLandmark.RIGHT_SHOULDER.value,
              "elbow": PoseLandmark.RIGHT_ELBOW.value, "wrist": PoseLandmark.RIGHT_WRIST.value},
    "left": {"hip": PoseLandmark.LEFT_HIP.value, "shoulder": PoseLandmark.LEFT_SHOULDER.value,
             "elbow": PoseLandmark.LEFT_ELBOW.value, "wrist": PoseLandmark.LEFT_WRIST.value},
}

# config.py exposes only the active position and anatomical-arm condition.
from config import load_settings, save_settings, set_position, set_arm, set_recording_timing, mode_dir


def make_landmarker(running_mode=vision.RunningMode.VIDEO):
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model: {MODEL_PATH}")
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=running_mode, num_poses=1)
    return vision.PoseLandmarker.create_from_options(opts)


def open_camera(index=0, width=None, height=None, verbose=True):
    """Windows webcam probing retained from your original nf_core.py."""
    backends = [("default", 0), ("MSMF", getattr(cv2, "CAP_MSMF", 1400)),
                ("DSHOW", getattr(cv2, "CAP_DSHOW", 700))]
    for name, api in backends:
        cap = cv2.VideoCapture(index) if api == 0 else cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            if verbose: print(f"  {name}: could not open")
            continue
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        brightest, best = 0.0, None
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None:
                brightness = float(frame.mean())
                if brightness > brightest:
                    brightest, best = brightness, frame
            time.sleep(0.03)
        if best is not None and brightest > 1.0:
            w, h = best.shape[1], best.shape[0]
            if verbose: print(f"  {name}: OK {w}x{h}")
            return cap, w, h
        cap.release()
        if verbose: print(f"  {name}: black/no frames")
    raise SystemExit("Cannot open webcam. Close other camera apps; check Windows camera permissions. Run diagnose_camera.py.")


def detect(landmarker, frame_bgr, timestamp_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(timestamp_ms))
    return result.pose_landmarks[0] if result.pose_landmarks else None


def to_pixels(lm, width, height):
    return (lm.x * width, lm.y * height)


def joints_usable(landmarks, ids, min_visibility):
    if landmarks is None:
        return False
    for idx in ids:
        lm = landmarks[idx]
        if (not math.isfinite(lm.x) or not math.isfinite(lm.y)
                or not math.isfinite(lm.visibility) or lm.visibility < min_visibility
                or not 0.0 <= lm.x <= 1.0 or not 0.0 <= lm.y <= 1.0):
            return False
    return True


def shoulder_angle(landmarks, arm, width, height, min_visibility):
    """(arm_elevation_degrees, usable); hip is deliberately NOT used."""
    ids = JOINTS[arm]
    if not joints_usable(landmarks, [ids["shoulder"], ids["elbow"]], min_visibility):
        return None, False
    a = arm_elevation(to_pixels(landmarks[ids["shoulder"]], width, height),
                      to_pixels(landmarks[ids["elbow"]], width, height))
    return a, a is not None


def shoulder_tilt(landmarks, width, height, min_visibility):
    """Both shoulders must be reliable; otherwise returns None."""
    r, l = JOINTS["right"]["shoulder"], JOINTS["left"]["shoulder"]
    if not joints_usable(landmarks, [r, l], min_visibility):
        return None
    return shoulder_line_tilt(to_pixels(landmarks[r], width, height),
                              to_pixels(landmarks[l], width, height))


def _screen_pt(lm, w, h, mirror):
    x = (1.0 - lm.x) if mirror else lm.x
    return int(x * w), int(lm.y * h)


def draw_skeleton(frame, landmarks, min_visibility=0.5, mirror=True):
    h, w = frame.shape[:2]
    pts = [_screen_pt(lm, w, h, mirror) for lm in landmarks]
    for a, b in CONNECTIONS:
        if joints_usable(landmarks, [a, b], min_visibility):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for lm, pt in zip(landmarks, pts):
        if math.isfinite(lm.x) and math.isfinite(lm.y) and 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
            color = (0, 0, 255) if lm.visibility >= min_visibility else (0, 165, 255)
            cv2.circle(frame, pt, 4, color, -1)


def highlight_joints(frame, landmarks, arm, mirror=True):
    """Highlight actual measurement joints, not the hip."""
    h, w = frame.shape[:2]
    for name in ("shoulder", "elbow"):
        lm = landmarks[JOINTS[arm][name]]
        if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1):
            continue
        p = _screen_pt(lm, w, h, mirror)
        cv2.circle(frame, p, 10, (255, 255, 0), 2)
        cv2.putText(frame, name, (p[0]+12, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)


def frame_quality(usable_flags, settings):
    """Return (acceptable, ratio, longest_bad_run). Missing frames stay missing."""
    n = len(usable_flags)
    if not n:
        return False, 0.0, 0
    ratio = sum(bool(x) for x in usable_flags) / n
    run = longest = 0
    for good in usable_flags:
        run = 0 if good else run + 1
        longest = max(longest, run)
    ok = ratio >= settings["min_usable_ratio"] and longest <= settings["max_bad_run"]
    return ok, ratio, longest


def dtw_distance(seq_a, seq_b):
    """Same original path-length-normalized DTW; input is 1D valid angle traces."""
    a = np.asarray(seq_a, dtype=float)
    b = np.asarray(seq_b, dtype=float)
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf"), 0
    n, m = len(a), len(b)
    cost = np.abs(a[:, None] - b[None, :])
    acc = np.full((n+1, m+1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            acc[i, j] = cost[i-1, j-1] + min(acc[i-1, j], acc[i, j-1], acc[i-1, j-1])
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        steps += 1
        choices = (acc[i-1, j-1], acc[i-1, j], acc[i, j-1])
        k = int(np.argmin(choices))
        if k == 0: i, j = i-1, j-1
        elif k == 1: i -= 1
        else: j -= 1
    steps += i+j
    return float(acc[n, m] / steps), steps


def decide(max_angle, dtw_score, settings, any_unusable):
    """Priority: unreliable -> below target -> DTW mismatch -> reached.
    If DTW isn't calibrated, show an ANGLE-only result, not a complete pass.
    """
    if any_unusable or max_angle is None:
        return "Unable to judge - move into view and repeat", (0, 165, 255)
    if max_angle < settings["movement_target_deg"] - settings["target_tolerance_deg"]:
        return "Arm elevation below target", (0, 0, 255)
    thr = settings["dtw_threshold"]
    if thr is None:
        return "Angle target reached (DTW not calibrated)", (0, 200, 200)
    if dtw_score is None:
        return "Unable to judge - no DTW score", (0, 165, 255)
    if dtw_score > thr:
        return "Movement differs from reference", (0, 0, 255)
    return "Target reached", (0, 200, 0)


def clip_timestamps(path, fps):
    """Use recorded per-frame monotonic times if present; otherwise nominal FPS.
    Returns list[int] or None. The fallback is an estimate, not measured time.
    """
    csv_path = Path(path).with_suffix(".timestamps.csv")
    if not csv_path.exists():
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [int(float(r["timestamp_ms"])) for r in rows]


def frame_timestamp(index, timestamps, fps, previous):
    raw = timestamps[index] if timestamps is not None and index < len(timestamps) else int(index*1000/fps)
    return max(int(raw), previous+1)
