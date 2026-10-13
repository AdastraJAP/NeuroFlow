"""
STEP 2 + 3 of the advisor's guide: check the joints, the angle, and the
coordinate representation.

Stand in your fixed camera position. Hold the arm (a) down at your side,
(b) halfway up, (c) at the target. Read the angle each time.

Expected:  arm down  ~0 deg     upper arm horizontal  ~90 deg

Coordinate representation: image landmarks converted to pixels.

Run:  python step2_check_angle.py
Keys: q quit    a switch arm
"""

import time

import cv2

import nf_core as nf


def main():
    s = nf.load_settings()
    arm = s["arm"]

    print("opening camera...")
    cap, w, h = nf.open_camera(0, s["capture_width"], s["capture_height"])
    print(f"capture resolution: {w} x {h}")
    print("coordinate representation: image landmarks -> pixels")

    win = "Step 2-3: angle check  |  q quit, a switch arm"
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

            # Show a mirrored picture, because that is what feels natural.
            frame = cv2.flip(raw, 1)

            if lms is not None:
                nf.draw_skeleton(frame, lms, s["min_visibility"], mirror=True)
                nf.highlight_joints(frame, lms, arm, mirror=True)
                angle, usable = nf.shoulder_angle(
                    lms, arm, w, h, s["min_visibility"]
                )
                if angle is not None:
                    txt = f"{arm} shoulder angle: {angle:5.1f} deg"
                    colour = (0, 255, 0) if usable else (0, 165, 255)
                    cv2.putText(frame, txt, (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
                    state = "joints usable" if usable else "JOINTS NOT RELIABLE"
                    cv2.putText(frame, state, (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            else:
                cv2.putText(frame, "no person detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.putText(frame, "arm down ~0 deg   horizontal ~90 deg",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("a"):
                arm = "left" if arm == "right" else "right"
                print(f"arm -> {arm} (your real {arm} arm)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
