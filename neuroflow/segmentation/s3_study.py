"""
B8: S3 study, automatic vs manual segmentation (owner: Member B).

Compares a segmenter's reps with reference reps (B9 manual labels on real clips;
the truth files on the mocks) block by block, then pools the blocks.

Matching: each predicted rep is paired with at most one reference rep, greedily
by highest IoU of their [t_start, t_end] windows; a pair counts as a hit when
IoU >= MATCH_IOU. An unknown boundary (NaN: the clip starts or ends before
the arm is at rest) is set to the clip's edge for matching only, so the rep
still counts; its timing error is not computed.

Per block:  n_pred, n_ref, count_err = n_pred - n_ref, hits, precision, recall,
            F1, mean IoU of hits, signed start/end errors of hits (pred - ref, ms).
Summary:    count MAE, exact-count rate, off-by-one rate (|count_err| == 1),
            pooled precision / recall / F1, boundary error (mean, SD, median |e|,
            share within +-TOLERANCE_MS) over boundaries known on both sides, also
            split into start and end, and how many were known; per segmenter.

MATCH_IOU = 0.5 is a common choice in event detection, not a value from our
sources; report sensitivity if results sit near it.

Run (from the repo root):
    python -m neuroflow.segmentation.s3_study                  # mocks vs truth
    python -m neuroflow.segmentation.s3_study --labels DIR --frames frames.parquet \
           --calib calibration_manifest.json --angle theta_v_deg
                                   # real clips: DIR/<block_id>_labels.csv; rest/rom come
                                   # only from a checked manifest (calib_manifest.py)
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from neuroflow.segmentation.segment import TOLERANCE_MS

MATCH_IOU = 0.5


# ---- 1. matching ----

def interval_iou(a_start, a_end, b_start, b_end):
    """IoU of two time windows; NaN if any boundary is unknown or a window is empty."""
    vals = np.array([a_start, a_end, b_start, b_end], dtype=float)
    if np.isnan(vals).any() or a_end <= a_start or b_end <= b_start:
        return np.nan
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return inter / (max(a_end, b_end) - min(a_start, b_start))


def _edges(reps, clip):
    """(starts, ends) with unknown boundaries set to the clip edges (if clip given)."""
    if len(reps) == 0:
        return np.empty(0), np.empty(0)
    s = reps["t_start_ms"].to_numpy(dtype=float)
    e = reps["t_end_ms"].to_numpy(dtype=float)
    if clip is not None:
        s = np.where(np.isnan(s), clip[0], s)
        e = np.where(np.isnan(e), clip[1], e)
    return s, e


def match_reps(pred, ref, min_iou=MATCH_IOU, clip=None):
    """
    One-to-one matches as a list of (pred_row, ref_row, iou), best IoU first.
    clip = (first, last) capture time of the block: unknown boundaries count as
    the clip edge for matching. Without clip, a rep with an unknown boundary
    can't be matched.
    """
    ps, pe = _edges(pred, clip)
    rs, re_ = _edges(ref, clip)
    pairs = []
    for i in range(len(pred)):
        for j in range(len(ref)):
            iou = interval_iou(ps[i], pe[i], rs[j], re_[j])
            if not np.isnan(iou) and iou >= min_iou:
                pairs.append((iou, i, j))
    used_p, used_r, out = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
            out.append((i, j, iou))
    return out


# ---- 2. metrics ----

def _prf(hits, n_pred, n_ref):
    precision = hits / n_pred if n_pred else np.nan
    recall = hits / n_ref if n_ref else np.nan
    f1 = 2 * hits / (n_pred + n_ref) if n_pred + n_ref else np.nan
    return precision, recall, f1


def block_metrics(pred, ref, min_iou=MATCH_IOU, clip=None):
    """(summary dict, per-hit errors DataFrame) for one block. Errors stay NaN
    where a boundary is unknown on either side."""
    m = match_reps(pred, ref, min_iou, clip)
    errs = pd.DataFrame([{
        "iou": iou,
        "err_start_ms": pred["t_start_ms"].iloc[i] - ref["t_start_ms"].iloc[j],
        "err_peak_ms": pred["t_peak_ms"].iloc[i] - ref["t_peak_ms"].iloc[j],
        "err_end_ms": pred["t_end_ms"].iloc[i] - ref["t_end_ms"].iloc[j],
    } for i, j, iou in m], columns=["iou", "err_start_ms", "err_peak_ms", "err_end_ms"])
    precision, recall, f1 = _prf(len(m), len(pred), len(ref))
    return {
        "n_pred": len(pred), "n_ref": len(ref), "count_err": len(pred) - len(ref),
        "hits": len(m), "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": errs["iou"].mean() if len(errs) else np.nan,
    }, errs


def study(blocks, min_iou=MATCH_IOU, tol_ms=TOLERANCE_MS):
    """
    blocks: iterable of (segmenter, block_id, pred_reps, ref_reps[, (clip_first_ms, clip_last_ms)]).
    Returns (per-block table, summary table with one row per segmenter).
    """
    rows, all_errs = [], []
    for seg, block, pred, ref, *clip in blocks:
        s, e = block_metrics(pred, ref, min_iou, clip[0] if clip else None)
        rows.append({"segmenter": seg, "block_id": block, **s})
        all_errs.append(e.assign(segmenter=seg, block_id=block))
    per_block = pd.DataFrame(rows)
    errs = pd.concat(all_errs, ignore_index=True) if all_errs else pd.DataFrame()

    summary = []
    for seg, b in per_block.groupby("segmenter", sort=False):
        e = errs[errs["segmenter"] == seg]
        bound = pd.concat([e["err_start_ms"], e["err_end_ms"]]).dropna().to_numpy()
        st, en = e["err_start_ms"].dropna().to_numpy(), e["err_end_ms"].dropna().to_numpy()
        precision, recall, f1 = _prf(b["hits"].sum(), b["n_pred"].sum(), b["n_ref"].sum())
        summary.append({
            "segmenter": seg, "blocks": len(b),
            "count_mae": b["count_err"].abs().mean(),
            "exact_count_rate": (b["count_err"] == 0).mean(),
            "off_by_one_rate": (b["count_err"].abs() == 1).mean(),
            "precision": precision, "recall": recall, "f1": f1,
            "bound_err_mean_ms": bound.mean() if len(bound) else np.nan,
            "bound_err_sd_ms": bound.std(ddof=1) if len(bound) > 1 else np.nan,
            "bound_err_median_abs_ms": np.median(np.abs(bound)) if len(bound) else np.nan,
            "within_tol_rate": (np.abs(bound) <= tol_ms).mean() if len(bound) else np.nan,
            "starts_known": f"{len(st)}/{len(e)}",
            "start_err_median_abs_ms": np.median(np.abs(st)) if len(st) else np.nan,
            "ends_known": f"{len(en)}/{len(e)}",
            "end_err_median_abs_ms": np.median(np.abs(en)) if len(en) else np.nan,
        })
    return per_block, pd.DataFrame(summary)


# ---- 3. runs ----

def mock_blocks():
    """Both segmenters on every mock, against the truth files (stand-in for manual labels)."""
    from neuroflow.segmentation.segment import calibration_refs, segment_fixed, segment_reps
    fx = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")
    calib = calibration_refs()
    for path in sorted(glob.glob(os.path.join(fx, "mock_frames_*_truth.csv"))):
        v = os.path.basename(path)[len("mock_frames_"):-len("_truth.csv")]
        frames = pd.read_parquet(os.path.join(fx, f"mock_frames_{v}.parquet"))
        truth = pd.read_csv(path)
        rest, rom = calib[frames["subject_id"].iloc[0]]
        clip = (frames["t_capture_ms"].min(), frames["t_capture_ms"].max())
        yield "v0_fixed", v, segment_fixed(frames), truth, clip
        yield "v1_rom", v, segment_reps(frames, rom, rest), truth, clip


def check_labels(ref, frames, path):
    """
    A labels table may only be compared with the clip it was made on, on the same
    clock: time_source must be t_capture_ms (not video_fps), and block_id,
    subject_id and session_id must match the frames of that block, and every
    marked time must lie inside the clip's capture interval [first, last
    t_capture_ms]. Raises otherwise. A file with 0 reps has no rows to check and passes.
    (Finite, ordered times are already checked by load_labels.)
    """
    bad_clock = sorted(set(ref["time_source"].dropna()) - {"t_capture_ms"})
    if bad_clock or ref["time_source"].isna().any():
        raise ValueError(f"{path}: time_source must be t_capture_ms, got "
                         f"{bad_clock or ['(missing)']}; re-label with --frames")
    for col in ("block_id", "subject_id", "session_id"):
        want = frames[col].astype(str).unique()
        if len(want) != 1:
            raise ValueError(f"frames for {path} hold several {col} values {sorted(want)}")
        got = sorted(set(ref[col].astype(str)) - {want[0]})
        if got:
            raise ValueError(f"{path}: {col} {got} doesn't match the frames ({want[0]})")
    T = ref[["t_start_ms", "t_peak_ms", "t_end_ms"]].to_numpy(dtype=float)
    lo, hi = frames["t_capture_ms"].min(), frames["t_capture_ms"].max()
    outside = ~np.isnan(T) & ((T < lo) | (T > hi))
    if outside.any():
        reps = ref.loc[outside.any(axis=1), "rep_idx"].tolist()
        raise ValueError(f"{path}: reps {reps} have times outside the clip "
                         f"({lo:.0f}-{hi:.0f} ms capture time)")


def real_blocks(frames, labels_dir, rest_ref, rom_ref, angle_col, allow_missing=False):
    """
    Both segmenters on each real clip, against its B9 labels file
    <labels_dir>/<block_id>_labels.csv (checked with check_labels).
    Every block in frames is expected to have a labels file. Missing ones are listed
    all at once and raise, unless allow_missing=True, which prints the list and
    leaves those blocks out.
    """
    from neuroflow.segmentation.label_tool import load_labels
    from neuroflow.segmentation.segment import segment_fixed, segment_reps
    blocks = list(frames.groupby("block_id"))
    paths = {b: os.path.join(labels_dir, f"{b}_labels.csv") for b, _ in blocks}
    missing = [paths[b] for b, _ in blocks if not os.path.exists(paths[b])]
    if missing:
        msg = f"{len(missing)} of {len(blocks)} expected label files are missing: {missing}"
        if not allow_missing:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}; those blocks are left out")
    for block, f in blocks:
        if not os.path.exists(paths[block]):
            continue
        ref = load_labels(paths[block])
        check_labels(ref, f, paths[block])
        clip = (f["t_capture_ms"].min(), f["t_capture_ms"].max())
        yield "v0_fixed", block, segment_fixed(f, angle_col=angle_col), ref, clip
        yield "v1_rom", block, segment_reps(f, rom_ref, rest_ref, angle_col=angle_col), ref, clip


def _print(per_block, summary):
    pd.set_option("display.width", 200)
    print(per_block.round(3).to_string(index=False))
    print()
    print(summary.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", help="folder with <block_id>_labels.csv from the B9 tool")
    p.add_argument("--frames", help="frames.parquet of the labelled clips")
    p.add_argument("--calib", help="calibration manifest (calib_manifest.py) for the real clips")
    p.add_argument("--angle", default="theta_ht_deg")
    p.add_argument("--allow-missing", action="store_true",
                   help="leave out blocks without a labels file (listed) instead of stopping")
    a = p.parse_args()

    # Hand check: IoU of [0, 10] and [5, 15] is 5 / 15.
    assert abs(interval_iou(0, 10, 5, 15) - 1 / 3) < 1e-12

    if a.labels:
        from neuroflow.segmentation.calib_manifest import check_manifest, load_manifest
        if not (a.frames and a.calib):
            p.error("--labels needs --frames and --calib")
        frames = pd.read_parquet(a.frames)
        m = load_manifest(a.calib)
        rest, rom, warnings = check_manifest(m, frames, a.angle)
        print(f"S3 study on real clips ({a.angle}, IoU >= {MATCH_IOU})")
        print(f"calibration: rest {rest:.3f}, rom {rom:.3f} deg from {m['source_blocks']} "
              f"(subject {m['subject_id']}, session {m['session_id']}, {a.calib})")
        for w in warnings:
            print(f"WARNING: {w}")
        print()
        _print(*study(real_blocks(frames, a.labels, rest, rom, a.angle, a.allow_missing)))
    else:
        print(f"S3 study on the mocks vs truth (IoU >= {MATCH_IOU}, tolerance +-{TOLERANCE_MS:.0f} ms)\n")
        _print(*study(mock_blocks()))


if __name__ == "__main__":
    main()
