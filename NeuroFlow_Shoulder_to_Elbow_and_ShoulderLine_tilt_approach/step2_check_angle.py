"""Live geometry check: hip-free arm elevation and shoulder-line tilt.

Run python step2_check_angle.py
Keys: q quit | a switch anatomical arm (also selects it in the menu) |
      b reset shoulder-tilt baseline | s save screenshot
Check 0, 45, 90 and 120 degrees with a goniometer/phone angle reference.
"""
import time
from datetime import datetime
from pathlib import Path
import cv2
import nf_core as nf
from bilateral import BilateralRepetitionDetector
from processing import measure_bilateral_frame


def main():
    s = nf.load_settings()
    arm = s["arm"]
    cap, w, h = nf.open_camera(s.get("camera_index", 0), s["capture_width"], s["capture_height"])
    win = "Neuro-Flow: hip-free angle  |  a arm  b baseline  s screenshot  q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1280, w), min(720, h))
    start, prev_ts, baseline = time.monotonic(), -1, None
    bilateral_detector = BilateralRepetitionDetector(s['rep_timeout_s'])
    preview_result_ts = None
    try:
        with nf.make_landmarker() as landmarker:
            while True:
                ok, raw = cap.read()
                if not ok: break
                ts = max(prev_ts+1, int((time.monotonic()-start)*1000))
                prev_ts = ts
                lms = nf.detect(landmarker, raw, ts)
                frame = cv2.flip(raw, 1)
                if arm == 'both':
                    if preview_result_ts is not None and ts - preview_result_ts >= 2000:
                        bilateral_detector = BilateralRepetitionDetector(s['rep_timeout_s'])
                        preview_result_ts = None
                    if lms is not None:
                        nf.draw_skeleton(frame, lms, s['min_visibility'], mirror=True)
                        for side in ('left', 'right'):
                            nf.highlight_joints(frame, lms, side, mirror=True)
                    row = measure_bilateral_frame(0, ts, lms, s, w, h)
                    if bilateral_detector.result is None:
                        if bilateral_detector.push(ts, row['left_arm_elevation_deg'],
                                                   row['right_arm_elevation_deg']) is not None:
                            preview_result_ts = ts
                    fmt = lambda value: f'{value:.1f}' if value is not None else '--'
                    cv2.putText(frame, f'Anatomical LEFT: {fmt(row["left_arm_elevation_deg"])} deg | {row["left_status"]}',
                                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2)
                    cv2.putText(frame, f'Anatomical RIGHT: {fmt(row["right_arm_elevation_deg"])} deg | {row["right_status"]}',
                                (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2)
                    cv2.putText(frame, f'LEFT - RIGHT: {fmt(row["elevation_diff_signed_deg"])} deg',
                                (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2)
                    cv2.putText(frame, bilateral_detector.instruction,
                                (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
                    cv2.putText(frame, f'Status: {bilateral_detector.state}',
                                (15, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,255), 2)
                    tilt = row['shoulder_tilt_deg']
                    if baseline is None and tilt is not None: baseline = tilt
                    delta = nf.tilt_change(tilt, baseline)
                    cv2.putText(frame, f'Shoulder-line change: {fmt(delta)} deg',
                                (15, 207), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                    cv2.putText(frame, 'Detector resets on arm change; recording is option 4/5/6/10/17',
                                (15, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                    cv2.imshow(win, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): break
                    if key == ord('a'):
                        arm = 'right'
                        s = nf.set_arm(arm)
                        baseline = None
                        bilateral_detector = BilateralRepetitionDetector(s['rep_timeout_s'])
                        preview_result_ts = None
                    if key == ord('b'): baseline = None
                    if key == ord('s'):
                        out = nf.mode_dir('review', s) / f'live_skeleton_{datetime.now():%Y%m%d_%H%M%S}.jpg'
                        out.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out), frame)
                    continue
                if lms is None:
                    cv2.putText(frame, "No person detected", (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)
                else:
                    nf.draw_skeleton(frame, lms, s["min_visibility"], mirror=True)
                    nf.highlight_joints(frame, lms, arm, mirror=True)
                    angle, usable = nf.shoulder_angle(lms, arm, w, h, s["min_visibility"])
                    tilt = nf.shoulder_tilt(lms, w, h, s["min_visibility"])
                    if baseline is None and tilt is not None: baseline = tilt
                    delta = nf.tilt_change(tilt, baseline)
                    color = (0, 255, 0) if usable else (0, 165, 255)
                    label = f"{arm} arm elevation: {angle:.1f} deg" if usable else "ARM JOINTS UNRELIABLE"
                    cv2.putText(frame, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    txt = f"Shoulder-line change: {delta:+.1f} deg" if delta is not None else "Shoulder-line change: unavailable"
                    cv2.putText(frame, txt, (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    hip = lms[nf.JOINTS[arm]["hip"]]
                    cv2.putText(frame, f"Hip visibility (diagnostic only): {hip.visibility:.2f}",
                                (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, "Arm down ~0  |  horizontal ~90  |  hip NOT used",
                            (15, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                if key == ord('a'):
                    arm = "left" if arm == "right" else ("both" if s['position'] == 'seated' else 'right')
                    s = nf.set_arm(arm)
                    baseline = None
                    bilateral_detector = BilateralRepetitionDetector(s['rep_timeout_s'])
                    preview_result_ts = None
                    print("Selected anatomical arm:", arm)
                if key == ord('b'):
                    baseline = None
                    print("Tilt baseline will reset on the next usable frame")
                if key == ord('s'):
                    out = nf.mode_dir("review", s) / f"live_skeleton_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), frame)
                    print("Saved screenshot:", out)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__": main()
