"""Your original Day 0 webcam skeleton demo, simplified to reuse nf_core.
Run: python day0_pose.py | s screenshot | q quit
"""
import time
from datetime import datetime
from pathlib import Path
import cv2
import nf_core as nf


def main():
    s = nf.load_settings()
    cap, w, h = nf.open_camera(s.get("camera_index", 0), s["capture_width"], s["capture_height"])
    window = "Neuro-Flow: live MediaPipe skeleton | s screenshot | q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(1280,w), min(720,h))
    start, prev = time.monotonic(), -1
    try:
        with nf.make_landmarker() as detector:
            while True:
                ok, raw = cap.read()
                if not ok: break
                ts = max(prev+1, int((time.monotonic()-start)*1000)); prev=ts
                lms = nf.detect(detector, raw, ts)
                shown = cv2.flip(raw, 1)
                if lms is not None:
                    nf.draw_skeleton(shown, lms, s["min_visibility"], mirror=True)
                    label = ('Bilateral shoulders/elbows (anatomical labels in option 2)'
                             if s['arm'] == 'both' else
                             f"Hip visibility: {lms[nf.JOINTS[s['arm']]['hip']].visibility:.2f} (diagnostic)")
                    cv2.putText(shown, label,
                                (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                else:
                    cv2.putText(shown, "No person detected", (15,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
                cv2.imshow(window, shown)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'): break
                if k == ord('s'):
                    path = nf.mode_dir('review', s) / f"pose_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), shown)
                    print('Screenshot saved:', path)
    finally:
        cap.release(); cv2.destroyAllWindows()


if __name__ == '__main__': main()
