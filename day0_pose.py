"""
Neuro-Flow -- Day 0
Open the webcam, run MediaPipe Pose Landmarker, draw the skeleton live.

Setup (once):
    pip install mediapipe opencv-python
    curl -L -o pose_landmarker_lite.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

Run:
    python day0_pose.py
Press  q  to quit.
"""

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

# 35 bone segments, as (start_id, end_id) pairs
CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]


def draw_skeleton(frame, landmarks):
    """Draw bones and joints onto a BGR frame. Landmark x/y are 0..1 fractions."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {CAMERA_INDEX}. "
            "Try CAMERA_INDEX = 1, or check camera permissions."
        )

    start = time.time()
    prev = start
    fps = 0.0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror, so moving left looks left

            # OpenCV gives BGR; MediaPipe wants RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamps must increase strictly in VIDEO mode
            ts_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                draw_skeleton(frame, lms)

                knee = lms[PoseLandmark.RIGHT_KNEE.value]
                cv2.putText(
                    frame,
                    f"R knee  x={knee.x:.2f} y={knee.y:.2f} vis={knee.visibility:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                )
            else:
                cv2.putText(frame, "no person detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Neuro-Flow Day 0  --  press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
