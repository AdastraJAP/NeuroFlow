"""
STEP 4 of the advisor's guide: record the reference, practice and test clips.

Three SEPARATE sets, as the advisor asked:
    reference  - one clear complete movement
    practice   - 3 complete + 3 clearly incomplete  (used to pick the threshold)
    test       - fresh clips recorded AFTER the settings are frozen

One clip = one repetition (raise and return). Stay still for about a second
before and after the movement so the boundary is easy to see.

Run:
    python step4_record.py reference ref_01
    python step4_record.py practice  complete_01
    python step4_record.py practice  incomplete_01
    python step4_record.py test      T01

Keys: SPACE start/stop recording    q quit
"""

import os
import sys
import time

import cv2

import nf_core as nf

VALID = ("reference", "practice", "test")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in VALID:
        raise SystemExit(f"usage: python step4_record.py [{'|'.join(VALID)}] <name>")

    category, name = sys.argv[1], sys.argv[2]
    s = nf.load_settings()

    out_dir = os.path.join("clips", category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.avi")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} already exists. Choose another name.")

    print("opening camera...")
    cap, w, h = nf.open_camera(0, s["capture_width"], s["capture_height"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0

    writer = None
    recording = False
    frames = 0

    win = f"Step 4: record {category}/{name}  |  SPACE start-stop, q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"recording to {out_path} at {w}x{h} @ {fps:.0f} fps")
    print("press SPACE to start, SPACE again to stop")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Record the TRUE camera image, so step5 sees the same orientation
        # the live demo detects on. Only the preview is mirrored.
        shown = cv2.flip(frame, 1)

        if recording:
            writer.write(frame)
            frames += 1
            cv2.circle(shown, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(shown, f"REC  {frames} frames", (55, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(shown, "ready - press SPACE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(win, shown)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            if not recording:
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
                )
                recording = True
                frames = 0
            else:
                recording = False
                writer.release()
                writer = None
                print(f"saved {out_path}  ({frames} frames, "
                      f"{frames / fps:.1f} s)")
                break
        if k == ord("q"):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
