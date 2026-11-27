"""Test camera framing and per-joint MediaPipe visibility.

python visibility_test.py shoulder  (hip reported but NOT required)
python visibility_test.py squat     (future extension, not calibrated here)
"""
import csv
import sys
import time
import cv2
import nf_core as nf


def main():
    exercise = sys.argv[1] if len(sys.argv)>1 else "shoulder"
    if exercise not in ("shoulder", "squat"):
        raise SystemExit("Use shoulder or squat")
    s = nf.load_settings()
    arm = s["arm"]
    if exercise == "shoulder":
        arms = ('left', 'right') if arm == 'both' else (arm,)
        required = [nf.JOINTS[side][joint] for side in arms for joint in ('shoulder', 'elbow')]
        observed = [nf.JOINTS[side][joint] for side in arms for joint in ('hip', 'wrist')]
        observed += [nf.JOINTS["left"]["shoulder"], nf.JOINTS["right"]["shoulder"]]
    else:
        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark as P
        required = [P.LEFT_HIP.value, P.RIGHT_HIP.value, P.LEFT_KNEE.value,
                    P.RIGHT_KNEE.value, P.LEFT_ANKLE.value, P.RIGHT_ANKLE.value]
        observed = []
    joints = list(dict.fromkeys(required + observed))
    cap, w, h = nf.open_camera(s.get("camera_index", 0), s["capture_width"], s["capture_height"])
    window = f"Visibility {exercise} | q finish"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w,1280), min(h,720))
    start, prev, rows = time.monotonic(), -1, []
    try:
        with nf.make_landmarker() as detector:
            while True:
                ok, raw = cap.read()
                if not ok: break
                ts = max(prev+1, int((time.monotonic()-start)*1000)); prev=ts
                lms = nf.detect(detector, raw, ts)
                good = nf.joints_usable(lms, required, s["min_visibility"])
                row = {"timestamp_ms": ts, "pose_detected": int(lms is not None), "required_joints_usable": int(good)}
                if arm == 'both' and exercise == 'shoulder':
                    for side in ('left', 'right'):
                        row[f'{side}_required_joints_usable'] = int(nf.joints_usable(
                            lms, [nf.JOINTS[side]['shoulder'], nf.JOINTS[side]['elbow']], s['min_visibility']))
                for j in joints:
                    row[f"joint_{j}_visibility"] = lms[j].visibility if lms else ""
                rows.append(row)
                shown = cv2.flip(raw,1)
                if lms is not None: nf.draw_skeleton(shown, lms, s["min_visibility"], mirror=True)
                cv2.putText(shown, "REQUIRED JOINTS OK" if good else "REQUIRED JOINTS UNRELIABLE",
                            (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0) if good else (0,165,255), 2)
                if exercise == "shoulder":
                    cv2.putText(shown, "Hip measured for diagnostics, NOT required for angle",
                                (15,75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                cv2.imshow(window, shown)
                if cv2.waitKey(1)&0xFF == ord('q'): break
    finally:
        cap.release(); cv2.destroyAllWindows()
    if not rows: print("No frames captured"); return
    path = nf.mode_dir("calibration", s) / f"visibility_{exercise}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"Required-joint usable frames: {sum(r['required_joints_usable'] for r in rows)}/{len(rows)}")
    print("Saved", path)


if __name__ == '__main__': main()
