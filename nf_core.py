"""
Neuro-Flow shared code.

Coordinate representation for this experiment: MediaPipe IMAGE landmarks,
converted to PIXELS (x * width, y * height) before any angle is computed.
World landmarks are NOT used. Do not mix the two.
"""

import json
import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmark,
    PoseLandmarksConnections,
)

MODEL_PATH = "pose_landmarker_lite.task"
SETTINGS_PATH = "settings.json"

CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

# Shoulder abduction angle: vertex at the SHOULDER, rays to HIP and ELBOW.
# Arm hanging at the side  -> about 0 degrees
# Upper arm horizontal     -> about 90 degrees
JOINTS = {
    "right": {
        "hip": PoseLandmark.RIGHT_HIP.value,
        "shoulder": PoseLandmark.RIGHT_SHOULDER.value,
        "elbow": PoseLandmark.RIGHT_ELBOW.value,
        "wrist": PoseLandmark.RIGHT_WRIST.value,
    },
    "left": {
        "hip": PoseLandmark.LEFT_HIP.value,
        "shoulder": PoseLandmark.LEFT_SHOULDER.value,
        "elbow": PoseLandmark.LEFT_ELBOW.value,
        "wrist": PoseLandmark.LEFT_WRIST.value,
    },
}

DEFAULT_SETTINGS = {
    "version": "v1",
    "exercise": "shoulder_abduction",
    "arm": "right",
    "coordinate_representation": "image_landmarks_converted_to_pixels",
    "camera_setup": "fixed front view, camera at chest height, mark on floor",
    "capture_width": 1280,
    "capture_height": 720,
    "movement_target_deg": 85.0,
    "target_tolerance_deg": 5.0,
    "min_visibility": 0.5,
    "reference_clip": "clips/reference/ref_01.avi",
    "dtw_threshold": None,
}


def load_settings(path=SETTINGS_PATH):
    if os.path.exists(path):
        with open(path) as f:
            s = dict(DEFAULT_SETTINGS)
            s.update(json.load(f))
            return s
    return dict(DEFAULT_SETTINGS)


def save_settings(settings, path=SETTINGS_PATH):
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)


def make_landmarker(running_mode=vision.RunningMode.VIDEO):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Missing {MODEL_PATH}. Download it into this folder first."
        )
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def open_camera(index=0, width=None, height=None, verbose=True):
    """
    Open the webcam reliably on Windows.

    Some backends open the device but hand back black frames. We try each
    backend, ask for a few frames, and keep the first one that returns a
    picture that is actually not black.

    Returns (cap, width, height).
    """
    backends = [
        ("default", 0),
        ("MSMF", getattr(cv2, "CAP_MSMF", 1400)),
        ("DSHOW", getattr(cv2, "CAP_DSHOW", 700)),
    ]

    for name, api in backends:
        cap = cv2.VideoCapture(index) if api == 0 else cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            if verbose:
                print(f"  {name}: could not open")
            continue

        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Let the sensor warm up; the first frames are often black.
        brightest = 0.0
        frame = None
        for _ in range(15):
            ok, f = cap.read()
            if ok and f is not None:
                m = float(f.mean())
                if m > brightest:
                    brightest, frame = m, f
            time.sleep(0.03)

        if frame is not None and brightest > 1.0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
            if verbose:
                print(f"  {name}: OK  {w}x{h}  (mean brightness {brightest:.1f})")
            return cap, w, h

        if verbose:
            print(f"  {name}: frames are black (mean {brightest:.1f})")
        cap.release()

    raise SystemExit(
        "Could not get a picture from the camera.\n"
        "  - close Zoom, Teams, OBS, the Camera app, or anything else using it\n"
        "  - check Settings > Privacy & security > Camera > let desktop apps access\n"
        "  - try a different camera index: python diagnose_camera.py"
    )


def detect(landmarker, frame_bgr, timestamp_ms):
    """Run the model on one BGR frame. Returns the landmark list or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if result.pose_landmarks:
        return result.pose_landmarks[0]
    return None


def to_pixels(lm, width, height):
    """Normalized landmark -> pixel coordinates. Do this BEFORE any angle."""
    return (lm.x * width, lm.y * height)


def angle_at(a_px, b_px, c_px):
    """Angle at vertex b, in degrees, from three pixel points."""
    bax, bay = a_px[0] - b_px[0], a_px[1] - b_px[1]
    bcx, bcy = c_px[0] - b_px[0], c_px[1] - b_px[1]
    na = math.hypot(bax, bay)
    nc = math.hypot(bcx, bcy)
    if na == 0 or nc == 0:
        return None
    cosv = (bax * bcx + bay * bcy) / (na * nc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def joints_usable(landmarks, ids, min_visibility):
    """True only if every required joint is confident AND inside the frame."""
    for j in ids:
        lm = landmarks[j]
        if lm.visibility < min_visibility:
            return False
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return False
    return True


def shoulder_angle(landmarks, arm, width, height, min_visibility):
    """
    Returns (angle_degrees, usable).
    usable is False when a required joint cannot be trusted.
    """
    ids = JOINTS[arm]
    required = [ids["hip"], ids["shoulder"], ids["elbow"]]
    usable = joints_usable(landmarks, required, min_visibility)
    hip = to_pixels(landmarks[ids["hip"]], width, height)
    sho = to_pixels(landmarks[ids["shoulder"]], width, height)
    elb = to_pixels(landmarks[ids["elbow"]], width, height)
    return angle_at(hip, sho, elb), usable


def _screen_pt(lm, w, h, mirror):
    """
    Where to DRAW a landmark.

    Detection always runs on the true (unmirrored) camera image, so that
    MediaPipe's LEFT_/RIGHT_ labels match the person's real left and right.
    The picture shown to the user is mirrored, so drawing must mirror x too.
    """
    x = (1.0 - lm.x) if mirror else lm.x
    return (int(x * w), int(lm.y * h))


def draw_skeleton(frame, landmarks, min_visibility=0.5, mirror=True):
    h, w = frame.shape[:2]
    pts = [_screen_pt(lm, w, h, mirror) for lm in landmarks]
    for s, e in CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
    for lm, p in zip(landmarks, pts):
        colour = (0, 0, 255) if lm.visibility >= min_visibility else (0, 165, 255)
        cv2.circle(frame, p, 4, colour, -1)


def highlight_joints(frame, landmarks, arm, mirror=True):
    """Mark the three points used for the shoulder angle."""
    h, w = frame.shape[:2]
    ids = JOINTS[arm]
    for name in ("hip", "shoulder", "elbow"):
        p = _screen_pt(landmarks[ids[name]], w, h, mirror)
        cv2.circle(frame, p, 10, (255, 255, 0), 2)
        cv2.putText(frame, name, (p[0] + 12, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


# ----------------------------------------------------------------------
# DTW
# ----------------------------------------------------------------------

def dtw_distance(seq_a, seq_b):
    """
    Dynamic Time Warping between two 1-D sequences.
    Returns (normalized_distance, path_length).
    Normalization is by warping-path length, so clips of different
    durations can be compared.
    """
    a = np.asarray(seq_a, dtype=float)
    b = np.asarray(seq_b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf"), 0

    cost = np.abs(a[:, None] - b[None, :])

    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )

    # Backtrack to measure the path length
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        steps += 1
        choices = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
        k = int(np.argmin(choices))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    steps += i + j

    return acc[n, m] / steps, steps


# ----------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------

def decide(max_angle, dtw_score, settings, any_unusable):
    """
    Returns (message, colour_bgr).
    Order follows the advisor's table: trust the landmarks first, then the
    measured angle, then the DTW score.
    """
    target = settings["movement_target_deg"]
    tol = settings["target_tolerance_deg"]
    thr = settings["dtw_threshold"]

    if any_unusable or max_angle is None:
        return "Unable to judge - adjust your position", (0, 165, 255)
    if max_angle < target - tol:
        return "Arm elevation below target", (0, 0, 255)
    if thr is not None and dtw_score is not None and dtw_score > thr:
        return "Movement differs from the reference", (0, 0, 255)
    return "Target reached", (0, 200, 0)
