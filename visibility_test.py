"""
Neuro-Flow -- framing / visibility test.

Measures how reliably MediaPipe tracks the joints YOUR exercises need,
under a specific camera setup. Run it once per setup you want to compare.

Usage:
    python visibility_test.py squat
    python visibility_test.py shoulder

Stand in position, perform the movement slowly for ~20 seconds, press q.
It prints per-joint stats and writes a CSV you can plot later.
"""

import csv
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmark,
    PoseLandmarksConnections,
)

MODEL_PATH = "pose_landmarker_lite.task"
CAMERA_INDEX = 0

# The joints each exercise actually depends on.
JOINT_SETS = {
    "squat": [
        PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
        PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE,
        PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE,
        PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
    ],
    "shoulder": [
        PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW,
        PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST,
        PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
    ],
}

GOOD_VIS = 0.5           # below this, treat the joint as unusable
CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]


def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
    for lm, (x, y) in zip(landmarks, pts):
        colour = (0, 0, 255) if lm.visibility >= GOOD_VIS else (0, 165, 255)
        cv2.circle(frame, (x, y), 4, colour, -1)


def main():
    exercise = sys.argv[1] if len(sys.argv) > 1 else "squat"
    if exercise not in JOINT_SETS:
        raise SystemExit(f"exercise must be one of: {', '.join(JOINT_SETS)}")

    tracked = JOINT_SETS[exercise]
    out_path = f"visibility_{exercise}.csv"

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}.")

    rows = []
    start = time.time()
    frames = 0
    frames_all_good = 0

    win = f"Framing test: {exercise}  --  press q to finish"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            frames += 1

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                draw_skeleton(frame, lms)

                row = {"t_ms": ts_ms}
                all_good = True
                for j in tracked:
                    lm = lms[j.value]
                    row[f"{j.name}_vis"] = round(lm.visibility, 4)
                    row[f"{j.name}_y"] = round(lm.y, 4)
                    in_frame = 0.0 <= lm.y <= 1.0 and 0.0 <= lm.x <= 1.0
                    if lm.visibility < GOOD_VIS or not in_frame:
                        all_good = False
                rows.append(row)
                if all_good:
                    frames_all_good += 1

                status = "ALL JOINTS OK" if all_good else "SOME JOINTS WEAK"
                colour = (0, 255, 0) if all_good else (0, 165, 255)
                cv2.putText(frame, status, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            else:
                cv2.putText(frame, "NO PERSON", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            pct = 100.0 * frames_all_good / max(frames, 1)
            cv2.putText(frame, f"usable frames: {pct:.1f}%", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if not rows:
        print("No pose detected at all -- check framing and lighting.")
        return

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== {exercise} ===")
    print(f"frames captured      : {frames}")
    print(f"frames with a pose   : {len(rows)}")
    print(f"frames all joints ok : {frames_all_good} ({100.0*frames_all_good/frames:.1f}%)")
    print(f"\nper-joint mean visibility (and % of frames above {GOOD_VIS}):")
    for j in tracked:
        key = f"{j.name}_vis"
        vals = [r[key] for r in rows]
        mean = sum(vals) / len(vals)
        above = 100.0 * sum(1 for v in vals if v >= GOOD_VIS) / len(vals)
        flag = "" if above > 90 else "   <-- weak"
        print(f"  {j.name:18} mean={mean:.3f}   above={above:5.1f}%{flag}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
