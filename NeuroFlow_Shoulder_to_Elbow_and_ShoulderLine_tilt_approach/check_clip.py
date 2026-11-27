"""Review one recorded clip: skeleton screenshots + angle/tilt traces.

Usage: python check_clip.py clips/reference/ref_01.avi
Outputs review/<position>/<clipname>/{start,lowest,highest,end}.jpg, angle_trace.png,
        tilt_trace.png, frames.csv
"""
import sys
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nf_core as nf
from processing import load_bilateral_live_pose, process_clip, save_frames_csv


def review_bilateral(path, settings):
    """Plot saved live measurements; do not rerun MediaPipe on captured AVI."""
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    result = load_bilateral_live_pose(path, settings, metadata)
    rows = result['rows']
    out = nf.mode_dir('review', settings) / (path.parent.name + '_' + path.stem)
    out.mkdir(parents=True, exist_ok=True)
    save_frames_csv(result, out / 'frames.csv')
    t = [row['timestamp_ms'] / 1000 for row in rows]
    plt.figure(figsize=(10, 5))
    for arm in ('left', 'right'):
        plt.plot(t, [row[f'{arm}_arm_elevation_deg'] if row[f'{arm}_usable'] else float('nan')
                     for row in rows], label=f'Anatomical {arm}')
    plt.xlabel('Measured capture time (s)')
    plt.ylabel('Upper-arm elevation (degrees)')
    plt.title(path.name + ' | bilateral capture')
    plt.legend(); plt.tight_layout(); plt.savefig(out / 'angle_trace.png', dpi=140); plt.close()
    plt.figure(figsize=(10, 3))
    plt.plot(t, [row['elevation_diff_signed_deg'] if row['elevation_diff_signed_deg'] is not None
                 else float('nan') for row in rows])
    plt.axhline(0, color='grey', linestyle='--')
    plt.xlabel('Measured capture time (s)')
    plt.ylabel('Left minus right (degrees)')
    plt.tight_layout(); plt.savefig(out / 'difference_trace.png', dpi=140); plt.close()
    print(f"Bilateral frames: {len(rows)} | {result['repetition'].outcome} ({result['repetition'].termination_reason})")
    print('Saved:', out)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python check_clip.py clips/reference/ref_01.avi")
    path = Path(sys.argv[1])
    s = nf.load_settings()
    if s['arm'] == 'both':
        review_bilateral(path, s)
        return
    result = process_clip(path, s, snapshots=True)
    out = nf.mode_dir("review", s) / (path.parent.name + "_" + path.stem)
    out.mkdir(parents=True, exist_ok=True)
    save_frames_csv(result, out / "frames.csv")
    for name, (i, angle, frame, lms) in result["snapshots"].items():
        nf.draw_skeleton(frame, lms, s["min_visibility"], mirror=False)
        nf.highlight_joints(frame, lms, s["arm"], mirror=False)
        cv2.putText(frame, f"{name} frame {i}: {angle:.1f} deg (hip-free)",
                    (12,38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imwrite(str(out / (name + ".jpg")), frame)
    rows = result["rows"]
    t = [r["timestamp_ms"] / 1000 for r in rows]
    y = [r["arm_elevation_deg"] if r["usable"] else float("nan") for r in rows]
    plt.figure(figsize=(9,4))
    plt.plot(t, y, label="Arm elevation (hip-free)")
    plt.axhline(s["movement_target_deg"], color="red", linestyle="--", label="target")
    plt.xlabel("Time (s)"); plt.ylabel("Angle (degrees)"); plt.title(path.name)
    plt.legend(); plt.tight_layout(); plt.savefig(out / "angle_trace.png", dpi=140); plt.close()
    plt.figure(figsize=(9,3))
    plt.plot(t, [r["tilt_change_deg"] if r["tilt_change_deg"] is not None else float("nan") for r in rows])
    plt.axhline(s["shoulder_tilt_warning_deg"], color="orange", linestyle="--")
    plt.axhline(-s["shoulder_tilt_warning_deg"], color="orange", linestyle="--")
    plt.xlabel("Time (s)"); plt.ylabel("Tilt change (degrees)"); plt.title("Shoulder line vs repetition start")
    plt.tight_layout(); plt.savefig(out / "tilt_trace.png", dpi=140); plt.close()
    print(f"Frames: {len(rows)} | Usable: {result['usable_ratio']:.1%} | longest missing run: {result['max_bad_run']}")
    print(f"Timestamp source: {result['timestamp_source']}")
    if result["angles"]:
        print(f"Arm elevation: {min(result['angles']):.1f} to {max(result['angles']):.1f} degrees")
    if result["peak_tilt_change"] is not None:
        print(f"Peak absolute shoulder-line tilt change: {result['peak_tilt_change']:.1f} deg (exploratory)")
    print("Quality:", "OK for prototype comparison" if result["quality"] else "UNABLE TO JUDGE; improve framing and repeat")
    print("Saved:", out)
    if not result["snapshots"]:
        print("No reliable arm landmarks: no skeleton screenshots could be saved.")


if __name__ == "__main__": main()
