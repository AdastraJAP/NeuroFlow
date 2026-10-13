"""
STEP 5 of the advisor's guide: turn clips into angle traces, compute DTW
scores against the reference, and choose a threshold.

The same angle definition, coordinate representation and normalization are
used for every clip.

Run:
    python step5_score.py                 # score the practice set
    python step5_score.py test            # score the test set
    python step5_score.py --freeze 8.0    # write a chosen threshold into settings

Outputs:
    traces/<name>.csv      angle per frame
    traces/angles.png      the angle traces, plotted together
"""

import csv
import glob
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nf_core as nf

TRACE_DIR = "traces"


def extract_trace(path, settings):
    """Run the model over a clip. Returns (angles, n_unusable, n_frames)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    angles, unusable, total = [], 0, 0
    i = 0
    with nf.make_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            ts = int(i * 1000.0 / fps)
            i += 1
            lms = nf.detect(landmarker, frame, ts)
            if lms is None:
                unusable += 1
                continue
            a, usable = nf.shoulder_angle(
                lms, settings["arm"], w, h, settings["min_visibility"]
            )
            if a is None or not usable:
                unusable += 1
                continue
            angles.append(a)
    cap.release()
    return angles, unusable, total


def save_trace(name, angles):
    os.makedirs(TRACE_DIR, exist_ok=True)
    with open(os.path.join(TRACE_DIR, f"{name}.csv"), "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["frame", "shoulder_angle_deg"])
        for i, a in enumerate(angles):
            wri.writerow([i, round(a, 2)])


def main():
    s = nf.load_settings()

    if len(sys.argv) > 2 and sys.argv[1] == "--freeze":
        s["dtw_threshold"] = float(sys.argv[2])
        nf.save_settings(s)
        print(f"frozen: dtw_threshold = {s['dtw_threshold']}")
        print("settings.json updated. Record FRESH clips for the test set now.")
        return

    which = sys.argv[1] if len(sys.argv) > 1 else "practice"

    ref_path = s["reference_clip"]
    if not os.path.exists(ref_path):
        raise SystemExit(f"reference clip not found: {ref_path}")

    print(f"reference: {ref_path}")
    ref_angles, ref_bad, ref_total = extract_trace(ref_path, s)
    if not ref_angles:
        raise SystemExit("no usable frames in the reference clip.")
    save_trace("reference", ref_angles)
    print(f"  {len(ref_angles)} usable / {ref_total} frames, "
          f"max angle {max(ref_angles):.1f} deg\n")

    clips = sorted(glob.glob(os.path.join("clips", which, "*.avi")))
    if not clips:
        raise SystemExit(f"no clips in clips/{which}/")

    plt.figure(figsize=(9, 5))
    plt.plot(ref_angles, linewidth=2.5, color="black", label="reference")

    print(f"{'clip':<22} {'usable':>8} {'max deg':>9} {'DTW':>8}  result")
    print("-" * 68)
    rows = []
    for path in clips:
        name = os.path.splitext(os.path.basename(path))[0]
        angles, bad, total = extract_trace(path, s)
        save_trace(name, angles)

        if not angles:
            print(f"{name:<22} {'0/'+str(total):>8} {'-':>9} {'-':>8}  "
                  f"Unable to judge")
            rows.append((name, None, None, "Unable to judge"))
            continue

        usable_ratio = len(angles) / max(total, 1)
        score, _ = nf.dtw_distance(ref_angles, angles)
        mx = max(angles)
        msg, _ = nf.decide(mx, score, s, usable_ratio < 0.8)
        print(f"{name:<22} {str(len(angles))+'/'+str(total):>8} "
              f"{mx:>9.1f} {score:>8.2f}  {msg}")
        rows.append((name, mx, score, msg))
        plt.plot(angles, alpha=0.8, label=name)

    plt.axhline(s["movement_target_deg"], color="red", linestyle="--",
                linewidth=1, label=f"target {s['movement_target_deg']:.0f} deg")
    plt.xlabel("frame")
    plt.ylabel("shoulder abduction angle (deg)")
    plt.title(f"Angle traces - {which} set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(TRACE_DIR, exist_ok=True)
    plot_path = os.path.join(TRACE_DIR, f"angles_{which}.png")
    plt.savefig(plot_path, dpi=130)
    print(f"\nplot saved: {plot_path}")

    scored = [r for r in rows if r[2] is not None]
    if which == "practice" and scored:
        comp = [r[2] for r in scored if "incomplete" not in r[0].lower()]
        inc = [r[2] for r in scored if "incomplete" in r[0].lower()]
        if comp and inc:
            print(f"\ncomplete   DTW: {min(comp):.2f} to {max(comp):.2f}")
            print(f"incomplete DTW: {min(inc):.2f} to {max(inc):.2f}")
            if max(comp) < min(inc):
                suggest = (max(comp) + min(inc)) / 2
                print(f"\nThe two groups separate. Suggested threshold: "
                      f"{suggest:.1f}")
                print(f"Freeze it with:  python step5_score.py --freeze "
                      f"{suggest:.1f}")
            else:
                print("\nThe groups OVERLAP. Do not tune settings until they "
                      "pass - inspect the clips and report this to the "
                      "advisor, it is a real finding.")


if __name__ == "__main__":
    main()
