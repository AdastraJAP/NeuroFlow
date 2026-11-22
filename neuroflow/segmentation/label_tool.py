"""
B9: manual labelling helper (owner: Member B).

Step through a clip and mark each rep's start, peak and end. The labels are
saved as a CSV with the same timing columns as the mock truth files, so B8 can
compare them with the segmenter using segment.validate(frames, labels, reps=...).

Labelling rule (team decision 2026-11-18, same as B1/B2): start and end are
where the arm passes **rest + 5 % of that rep's range** on the way up and on the
way down (about 4 deg into a 80 deg rep), NOT the first visible movement.
The peak is the highest point. If you can't tell, leave the rep out rather than guess.

Times: if the clip's frames.parquet is given, video frame i gets that file's
t_capture_ms for frame_idx i (the capture clock, the same clock the segmenter
uses). Without it, the tool falls back to i * 1000 / fps from the video header
and writes time_source = "video_fps". Those times are not on the capture clock,
so don't compare them with segmenter output.

Run:
    python -m neuroflow.segmentation.label_tool CLIP [--frames frames.parquet]
           [--out labels.csv] [--labeler NAME] [--angle theta_v_deg] [--block T_01]
    --block picks one clip when frames.parquet holds several (e.g. Adastra's G1 file).
    python -m neuroflow.segmentation.label_tool        (no clip: logic self-test)

Keys:  a / d  back / forward 1 frame     j / l  back / forward 10 frames
       space  play / pause               m      mark the next point (start -> peak -> end)
       u      undo the last mark         s      save
       q / Esc / Enter, or close the window   save and quit
Mouse: click the timeline strip at the bottom to jump to that moment.
"""

import argparse
import os

import numpy as np
import pandas as pd

KINDS = ("start", "peak", "end")
LABEL_COLUMNS = ["rep_idx", "t_start_ms", "t_peak_ms", "t_end_ms",
                 "subject_id", "session_id", "block_id", "labeler", "time_source"]


# ---- 1. label bookkeeping (no GUI, fully testable) ----

class Labels:
    """Marks in time order: start, peak, end, start, peak, end, ..."""

    def __init__(self):
        self.marks = []   # list of t_ms, kind = KINDS[index % 3]

    def next_kind(self):
        return KINDS[len(self.marks) % 3]

    def mark(self, t_ms):
        """Add the next mark; it must come after the previous one. Returns its kind."""
        if not np.isfinite(t_ms):
            raise ValueError(f"{self.next_kind()} time {t_ms} is not a finite number")
        if self.marks and t_ms <= self.marks[-1]:
            raise ValueError(f"{self.next_kind()} at {t_ms:.0f} ms is not after the "
                             f"previous mark at {self.marks[-1]:.0f} ms")
        kind = self.next_kind()
        self.marks.append(float(t_ms))
        return kind

    def undo(self):
        """Remove the last mark; returns (kind, t_ms) or None if there was none."""
        if not self.marks:
            return None
        t = self.marks.pop()
        return KINDS[len(self.marks) % 3], t

    def to_frame(self, subject_id="", session_id="", block_id="", labeler="",
                 time_source="t_capture_ms"):
        """Reps table; an unfinished last rep keeps NaN for the points not yet marked."""
        rows = []
        for i in range(0, len(self.marks), 3):
            m = self.marks[i:i + 3] + [np.nan] * (3 - len(self.marks[i:i + 3]))
            rows.append({"rep_idx": i // 3, "t_start_ms": m[0], "t_peak_ms": m[1], "t_end_ms": m[2],
                         "subject_id": subject_id, "session_id": session_id, "block_id": block_id,
                         "labeler": labeler, "time_source": time_source})
        return pd.DataFrame(rows, columns=LABEL_COLUMNS)


# ---- 2. frame index -> time ----

def frame_times(n_frames, frames=None, fps=None):
    """
    (t_ms per video frame, time_source).
    frames: the clip's frames.parquet table; video frame i -> t_capture_ms of frame_idx i.
    Otherwise fps from the video header: i * 1000 / fps (flagged "video_fps").
    """
    if frames is not None:
        # Strict 1:1 mapping, so no video frame ever gets a missing or wrong time:
        # frame_idx must be exactly 0 .. n_frames-1 (unique, no gaps, same count as
        # the decoded video), and the times finite and strictly increasing.
        idx = frames["frame_idx"].to_numpy()
        if not frames["frame_idx"].is_unique:
            raise ValueError("frames.parquet has duplicate frame_idx values")
        if len(idx) != n_frames or set(idx.tolist()) != set(range(n_frames)):
            raise ValueError(f"frames.parquet frame_idx must be exactly 0..{n_frames - 1} "
                             f"(the decoded video has {n_frames} frames); got {len(idx)} rows, "
                             f"frame_idx {idx.min()}..{idx.max()}")
        t = frames.set_index("frame_idx")["t_capture_ms"].sort_index().to_numpy(dtype=float)
        if not np.isfinite(t).all():
            raise ValueError("frames.parquet has missing or non-finite t_capture_ms")
        if not (np.diff(t) > 0).all():
            raise ValueError("frames.parquet t_capture_ms is not strictly increasing in frame_idx")
        return t, "t_capture_ms"
    if not fps or fps <= 0:
        raise ValueError("no frames.parquet and no usable fps in the video header")
    return np.arange(n_frames) * 1000.0 / fps, "video_fps"


def select_block(frames, block_id=None):
    """Frames of one clip. A file with several block_ids needs block_id; one with a
    single block can omit it. Raises with the available ids otherwise."""
    blocks = sorted(frames["block_id"].astype(str).unique()) if "block_id" in frames.columns else []
    if block_id is None:
        if len(blocks) > 1:
            raise ValueError(f"frames hold several clips {blocks}; pass --block")
        return frames
    if block_id not in blocks:
        raise ValueError(f"block {block_id!r} not in frames; available: {blocks}")
    return frames[frames["block_id"].astype(str) == block_id]


# ---- 3. save / load (load is what B8 uses) ----

def save_labels(df, path):
    df.to_csv(path, index=False)


def load_labels(path, complete_only=True):
    """
    Read a labels CSV and check it: times are numbers and finite (NaN only for a
    point not marked yet, never inf), start < peak < end inside each rep (also
    for the points of an unfinished rep), reps in order and not overlapping.
    complete_only drops reps with a missing point.
    Returns columns in LABEL_COLUMNS order, rep_idx renumbered from 0.
    Whether the times lie inside the clip is checked against its frames in B8
    (s3_study.check_labels).
    """
    # ids as text, so e.g. session "001" isn't read back as the number 1
    df = pd.read_csv(path, dtype={c: str for c in ("subject_id", "session_id", "block_id",
                                                   "labeler", "time_source")})
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[LABEL_COLUMNS].copy()
    times = ["t_start_ms", "t_peak_ms", "t_end_ms"]
    for c in times:
        num = pd.to_numeric(df[c], errors="coerce")
        if (num.isna() & df[c].notna()).any():
            raise ValueError(f"{path}: {c} has values that are not numbers")
        df[c] = num
    T = df[times].to_numpy(dtype=float)
    if np.isinf(T).any():
        raise ValueError(f"{path}: reps {df.loc[np.isinf(T).any(axis=1), 'rep_idx'].tolist()} "
                         "have infinite times")
    for i, row in enumerate(T):
        marked = row[~np.isnan(row)]
        if (np.diff(marked) <= 0).any():
            raise ValueError(f"{path}: reps [{df['rep_idx'].iloc[i]}] are not start < peak < end")
    if complete_only:
        df = df.dropna(subset=["t_start_ms", "t_peak_ms", "t_end_ms"])
    df = df.sort_values("t_start_ms").reset_index(drop=True)
    done = df.dropna(subset=["t_start_ms", "t_peak_ms", "t_end_ms"])
    bad = done[~((done["t_start_ms"] < done["t_peak_ms"]) & (done["t_peak_ms"] < done["t_end_ms"]))]
    if len(bad):
        raise ValueError(f"{path}: reps {bad['rep_idx'].tolist()} are not start < peak < end")
    if (done["t_start_ms"].to_numpy()[1:] <= done["t_end_ms"].to_numpy()[:-1]).any():
        raise ValueError(f"{path}: reps overlap")
    df["rep_idx"] = np.arange(len(df))
    return df


# ---- 4. GUI (OpenCV window + mouse callback) ----

STRIP_H = 80
COLOURS = {"start": (0, 200, 0), "peak": (0, 200, 255), "end": (0, 0, 230)}  # BGR


def _draw_strip(width, t, labels, cur, trace=None):
    """Timeline strip: optional angle trace, marks, and a cursor at frame cur."""
    import cv2
    strip = np.full((STRIP_H, width, 3), 40, np.uint8)
    t0, t1 = np.nanmin(t), np.nanmax(t)
    x_of = lambda tm: int((tm - t0) / max(t1 - t0, 1e-9) * (width - 1))
    if trace is not None and np.isfinite(trace).any():
        lo, hi = np.nanmin(trace), np.nanmax(trace)
        pts = [(x_of(tm), int(STRIP_H - 6 - (v - lo) / max(hi - lo, 1e-9) * (STRIP_H - 12)))
               for tm, v in zip(t, trace) if np.isfinite(tm) and np.isfinite(v)]
        if len(pts) > 1:
            cv2.polylines(strip, [np.array(pts, np.int32)], False, (170, 170, 170), 1)
    for i, tm in enumerate(labels.marks):
        cv2.line(strip, (x_of(tm), 0), (x_of(tm), STRIP_H), COLOURS[KINDS[i % 3]], 2)
    if np.isfinite(t[cur]):
        cv2.line(strip, (x_of(t[cur]), 0), (x_of(t[cur]), STRIP_H), (255, 255, 255), 1)
    return strip


def run_gui(clip, out, frames_path=None, labeler="", angle_col="theta_ht_deg", block_id=None):
    import cv2
    cap = cv2.VideoCapture(clip)
    video = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        video.append(img)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not video:
        raise SystemExit(f"could not read any frame from {clip}")

    frames = select_block(pd.read_parquet(frames_path), block_id) if frames_path else None
    t, source = frame_times(len(video), frames, fps)
    trace = None
    if frames is not None and angle_col in frames.columns:
        trace = frames.set_index("frame_idx")[angle_col].reindex(np.arange(len(video))).to_numpy()
    meta = {k: (str(frames[k].iloc[0]) if frames is not None and k in frames.columns else "")
            for k in ("subject_id", "session_id", "block_id")}
    if source == "video_fps":
        print("WARNING: no frames.parquet, times come from the video fps, not the capture clock")

    labels = Labels()
    if os.path.exists(out):   # resume an earlier session on this clip
        for _, r in pd.read_csv(out).iterrows():
            for c in ("t_start_ms", "t_peak_ms", "t_end_ms"):
                if np.isfinite(r[c]):
                    labels.mark(r[c])
        print(f"resumed {len(labels.marks)} marks from {out}")

    QUIT_KEYS = (ord("q"), ord("Q"), 27, 13, 10)  # q, Esc, Enter (CR / LF)
    h, w = video[0].shape[:2]
    state = {"cur": 0, "play": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and y >= h:
            target = np.nanmin(t) + x / (w - 1) * (np.nanmax(t) - np.nanmin(t))
            state["cur"] = int(np.nanargmin(np.abs(t - target)))

    def save():
        save_labels(labels.to_frame(labeler=labeler, time_source=source, **meta), out)
        print(f"saved {len(labels.marks) // 3} complete reps to {out}")

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", on_mouse)
    msg = ""
    while True:
        cur = state["cur"]
        img = video[cur].copy()
        info = f"frame {cur}/{len(video) - 1}  t={t[cur]:.0f} ms  next: {labels.next_kind()}  {msg}"
        cv2.putText(img, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("label", np.vstack([img, _draw_strip(w, t, labels, cur, trace)]))
        key = cv2.waitKey(int(1000 / (fps or 30)) if state["play"] else 30) & 0xFF
        step = {ord("a"): -1, ord("d"): 1, ord("j"): -10, ord("l"): 10}.get(key, 0)
        if state["play"] and key == 255:
            step = 1
        if step:
            state["cur"] = int(np.clip(cur + step, 0, len(video) - 1))
        elif key == ord(" "):
            state["play"] = not state["play"]
        elif key == ord("m"):
            try:
                msg = f"marked {labels.mark(t[cur])}"
            except ValueError as e:
                msg = str(e)
        elif key == ord("u"):
            undone = labels.undo()
            msg = f"undid {undone[0]}" if undone else "nothing to undo"
        elif key in (ord("s"), ord("S")):
            save()
            msg = "saved"
        elif key in QUIT_KEYS or cv2.getWindowProperty("label", cv2.WND_PROP_VISIBLE) < 1:
            save()
            break
    cv2.destroyAllWindows()


# ---- self-test ----

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", nargs="?")
    p.add_argument("--frames", help="the clip's frames.parquet (gives capture-clock times)")
    p.add_argument("--out", help="labels CSV (default: <clip>_labels.csv)")
    p.add_argument("--labeler", default="")
    p.add_argument("--angle", default="theta_ht_deg", help="angle shown on the timeline (theta_v_deg for V4 clips)")
    p.add_argument("--block", help="block_id of this clip when frames.parquet holds several")
    a = p.parse_args()
    if a.clip:
        run_gui(a.clip, a.out or os.path.splitext(a.clip)[0] + "_labels.csv", a.frames, a.labeler,
                a.angle, a.block)
        return

    # No clip: check the bookkeeping on a mock's truth times.
    import tempfile
    fx = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")
    truth = pd.read_csv(os.path.join(fx, "mock_frames_normal_truth.csv"))
    lab = Labels()
    for r in truth.itertuples():
        assert [lab.mark(r.t_start_ms), lab.mark(r.t_peak_ms), lab.mark(r.t_end_ms)] == list(KINDS)
    path = os.path.join(tempfile.mkdtemp(), "labels.csv")
    save_labels(lab.to_frame(subject_id="mock01", labeler="selftest"), path)
    back = load_labels(path)
    assert np.allclose(back[["t_start_ms", "t_peak_ms", "t_end_ms"]],
                       truth[["t_start_ms", "t_peak_ms", "t_end_ms"]])
    print(f"self-test passed: {len(back)} reps round-tripped through {path}")


if __name__ == "__main__":
    main()
