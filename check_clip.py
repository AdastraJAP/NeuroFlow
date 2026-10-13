"""
Inspect one recorded clip.

Confirms the file is readable, that the required landmarks stay usable, and
that the movement was actually captured. This is the advisor's Step 4 check:
"Replay it and confirm that all required landmarks remain usable."

Run:
    python check_clip.py clips/reference/ref_01.avi

Writes into review/<clipname>/ :
    start.jpg  lowest.jpg  highest.jpg  end.jpg   (skeleton drawn on)
    trace.png                                     (the angle over time)
"""

import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nf_core as nf


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python check_clip.py <path to .avi>")
    path = sys.argv[1]
    if not os.path.exists(path):
        raise SystemExit(f"file not found: {path}")

    s = nf.load_settings()
    arm = s["arm"]
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join("review", name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(
            f"OpenCV could not open {path}.\n"
            "The file is probably corrupt - record it again."
        )

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"file      : {path}")
    print(f"size      : {os.path.getsize(path)/1e6:.1f} MB")
    print(f"resolution: {w} x {h} @ {fps:.0f} fps")

    angles, frames, idx = [], [], 0
    n_no_person, n_unusable = 0, 0

    with nf.make_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = int(idx * 1000.0 / fps)
            lms = nf.detect(landmarker, frame, ts)
            idx += 1

            if lms is None:
                n_no_person += 1
                continue
            a, usable = nf.shoulder_angle(lms, arm, w, h, s["min_visibility"])
            if a is None or not usable:
                n_unusable += 1
                continue
            angles.append(a)
            frames.append((idx - 1, a, frame.copy(), lms))
    cap.release()

    total = idx
    print(f"frames    : {total}")
    if total == 0:
        raise SystemExit("The clip has no frames. Record it again.")

    print(f"no person : {n_no_person}")
    print(f"unusable  : {n_unusable}")
    print(f"usable    : {len(angles)}  ({100*len(angles)/total:.1f}%)")

    if not angles:
        raise SystemExit(
            "\nNo usable frames. Check framing: hip, shoulder and elbow on the "
            f"{arm} side must stay in view for the whole movement."
        )

    lo = min(angles)
    hi = max(angles)
    print(f"\nangle range: {lo:.1f} to {hi:.1f} deg   (moved {hi-lo:.1f} deg)")

    # Save representative frames with the skeleton drawn on (not mirrored:
    # these are the true images the model actually saw).
    picks = {
        "start": frames[0],
        "lowest": min(frames, key=lambda f: f[1]),
        "highest": max(frames, key=lambda f: f[1]),
        "end": frames[-1],
    }
    for label, (i, a, img, lms) in picks.items():
        shot = img.copy()
        nf.draw_skeleton(shot, lms, s["min_visibility"], mirror=False)
        nf.highlight_joints(shot, lms, arm, mirror=False)
        cv2.putText(shot, f"{label}  frame {i}  {a:.1f} deg", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"{label}.jpg"), shot)

    plt.figure(figsize=(8, 4))
    plt.plot(angles, linewidth=2)
    plt.axhline(s["movement_target_deg"], color="red", linestyle="--",
                label=f"target {s['movement_target_deg']:.0f} deg")
    plt.xlabel("usable frame")
    plt.ylabel(f"{arm} shoulder angle (deg)")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trace.png"), dpi=130)

    print(f"\nwrote {out_dir}/  -> start.jpg lowest.jpg highest.jpg end.jpg trace.png")
    print("Open those images. Check the circles sit on the correct joints,")
    print("and that 'highest.jpg' really shows the top of your movement.")

    # Plain-language verdict
    print()
    if len(angles) / total < 0.9:
        print("WARNING: more than 10% of frames were unusable. Fix the framing "
              "and record again.")
    elif hi - lo < 30:
        print("WARNING: the angle barely changed. Did the movement get "
              "recorded? Check highest.jpg.")
    else:
        print("This clip looks good.")


if __name__ == "__main__":
    main()
