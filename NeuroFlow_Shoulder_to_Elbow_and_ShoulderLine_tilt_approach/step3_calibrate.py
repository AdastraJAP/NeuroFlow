"""Simple phone-goniometer comparison at 0/45/90/120 degrees.

Hold the pose measured independently using a phone angle tool/goniometer,
then press 1=0, 2=45, 3=90, 4=120. This does NOT prove clinical accuracy.
"""
import csv
import time
from datetime import datetime
from pathlib import Path
import cv2
import nf_core as nf

TARGETS = {ord('1'): 0, ord('2'): 45, ord('3'): 90, ord('4'): 120}


def main():
    s = nf.load_settings()
    if s['arm'] == 'both':
        raise SystemExit('Angle cross-check records one anatomical arm at a time; select left or right first.')
    cap, w, h = nf.open_camera(s.get("camera_index", 0), s["capture_width"], s["capture_height"])
    win = "Angle cross-check: 1=0  2=45  3=90  4=120  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w,1280), min(h,720))
    path = nf.mode_dir("calibration", s) / "angle_checks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    start, prev = time.monotonic(), -1
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["time", "position", "arm", "target_from_external_tool_deg", "mediapipe_2d_deg", "abs_error_deg", "usable"])
            with nf.make_landmarker() as detector:
                while True:
                    ok, raw = cap.read()
                    if not ok: break
                    ts = max(prev+1, int((time.monotonic()-start)*1000)); prev=ts
                    lms = nf.detect(detector, raw, ts)
                    angle, usable = nf.shoulder_angle(lms, s["arm"], w, h, s["min_visibility"])
                    shown = cv2.flip(raw, 1)
                    if lms is not None: nf.draw_skeleton(shown, lms, s["min_visibility"], mirror=True)
                    cv2.putText(shown, f"Arm elevation: {angle:.1f} deg" if usable else "Joints unreliable",
                                (12,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if usable else (0,165,255), 2)
                    cv2.putText(shown, "Match a real angle tool; press 1=0 2=45 3=90 4=120",
                                (12,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.imshow(win, shown)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'): break
                    if k in TARGETS:
                        target = TARGETS[k]
                        error = abs(angle-target) if usable else None
                        writer.writerow([datetime.now().isoformat(timespec="seconds"), s["position"], s["arm"], target,
                                         round(angle,2) if usable else "", round(error,2) if error is not None else "", int(usable)])
                        f.flush()
                        print(f"External {target} deg -> model {angle if usable else 'unreliable'} | abs error {error}")
    finally:
        cap.release(); cv2.destroyAllWindows()
    print("Saved:", path)


if __name__ == '__main__': main()
