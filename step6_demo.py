"""
STEP 6 of the advisor's guide: the live demonstration.

Shows, on one screen: the landmarks, the live shoulder angle, the DTW score
against the frozen reference, and the feedback message.

Repetition boundaries are marked MANUALLY with the SPACE key. The advisor
allows this for the first demonstration - say so when you present it.
Automatic splitting is the next step.

Run:  python step6_demo.py
Keys: SPACE start/stop a repetition    q quit
"""

import csv
import os
import time
from datetime import datetime

import cv2

import nf_core as nf

LOG_PATH = "test_log.csv"


def append_log(row):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "max_angle_deg", "dtw_score",
                        "usable_ratio", "system_result"])
        w.writerow(row)


def load_reference(settings):
    path = os.path.join("traces", "reference.csv")
    if not os.path.exists(path):
        raise SystemExit(
            "traces/reference.csv not found. Run step5_score.py first."
        )
    with open(path) as f:
        return [float(r["shoulder_angle_deg"]) for r in csv.DictReader(f)]


def main():
    s = nf.load_settings()
    if s["dtw_threshold"] is None:
        print("WARNING: dtw_threshold is not frozen yet. The DTW check will "
              "be skipped.\n         Freeze it with step5_score.py --freeze\n")
    ref = load_reference(s)
    arm = s["arm"]

    print("opening camera...")
    cap, w, h = nf.open_camera(0, s["capture_width"], s["capture_height"])

    capturing = False
    angles, n_frames, n_bad = [], 0, 0
    message, msg_colour = "press SPACE to start a repetition", (255, 255, 255)
    last_score = None

    win = "Neuro-Flow demo  |  SPACE = start/stop repetition, q = quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    start = time.time()

    with nf.make_landmarker() as landmarker:
        while True:
            ok, raw = cap.read()
            if not ok:
                break

            # Detect on the TRUE image so left/right labels are anatomical.
            ts = int((time.time() - start) * 1000)
            lms = nf.detect(landmarker, raw, ts)

            # Show a mirrored picture.
            frame = cv2.flip(raw, 1)

            angle, usable = None, False
            if lms is not None:
                nf.draw_skeleton(frame, lms, s["min_visibility"], mirror=True)
                nf.highlight_joints(frame, lms, arm, mirror=True)
                angle, usable = nf.shoulder_angle(
                    lms, arm, w, h, s["min_visibility"]
                )

            if capturing:
                n_frames += 1
                if angle is not None and usable:
                    angles.append(angle)
                else:
                    n_bad += 1

            # ---- overlay ----
            if angle is not None:
                c = (0, 255, 0) if usable else (0, 165, 255)
                cv2.putText(frame, f"angle {angle:5.1f} deg", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
            cv2.putText(frame, f"target {s['movement_target_deg']:.0f} deg",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)
            if s["dtw_threshold"] is not None:
                cv2.putText(frame, f"DTW threshold {s['dtw_threshold']:.1f}",
                            (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1)
            if last_score is not None:
                cv2.putText(frame, f"last DTW {last_score:.2f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if capturing:
                cv2.circle(frame, (w - 40, 36), 12, (0, 0, 255), -1)
                peak = max(angles) if angles else 0.0
                cv2.putText(frame, f"recording rep   peak {peak:5.1f}",
                            (w - 330, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, message, (14, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, msg_colour, 2)

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF

            if k == ord(" "):
                if not capturing:
                    capturing = True
                    angles, n_frames, n_bad = [], 0, 0
                    message, msg_colour = "recording...", (255, 255, 255)
                else:
                    capturing = False
                    ratio = len(angles) / max(n_frames, 1)
                    if angles:
                        score, _ = nf.dtw_distance(ref, angles)
                        mx = max(angles)
                    else:
                        score, mx = None, None
                    message, msg_colour = nf.decide(mx, score, s, ratio < 0.8)
                    last_score = score
                    append_log([
                        datetime.now().strftime("%H:%M:%S"),
                        f"{mx:.1f}" if mx is not None else "",
                        f"{score:.2f}" if score is not None else "",
                        f"{ratio:.2f}",
                        message,
                    ])
                    print(f"rep: max {mx if mx is None else round(mx,1)} deg, "
                          f"DTW {score if score is None else round(score,2)}, "
                          f"usable {ratio:.0%}  ->  {message}")
            if k == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nlog written to {LOG_PATH}")


if __name__ == "__main__":
    main()
