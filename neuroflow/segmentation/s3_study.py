"""
B8: S3 study, automatic vs manual segmentation (owner: Member B).

Compares a segmenter's reps with reference reps (B9 manual labels on real clips;
the truth files on the mocks) block by block, then pools the blocks.

Matching: each predicted rep is paired with at most one reference rep, greedily
by highest IoU of their [t_start, t_end] windows; a pair counts as a hit when
IoU >= MATCH_IOU. Reps with an unknown boundary can't be matched.

Per block:  n_pred, n_ref, count_err = n_pred - n_ref, hits, precision, recall,
            F1, mean IoU of hits, signed start/end errors of hits (pred - ref, ms).
Summary:    count MAE, exact-count rate, off-by-one rate (|count_err| == 1),
            pooled precision / recall / F1, boundary error (mean, SD, median |e|,
            share within +-TOLERANCE_MS), per segmenter.

MATCH_IOU = 0.5 is a common choice in event detection, not a value from our
sources; report sensitivity if results sit near it.

Run (from the repo root):
    python -m neuroflow.segmentation.s3_study                  # mocks vs truth
    python -m neuroflow.segmentation.s3_study --labels DIR     # real clips: DIR/<block_id>_labels.csv
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


def match_reps(pred, ref, min_iou=MATCH_IOU):
    """One-to-one matches as a list of (pred_row, ref_row, iou), best IoU first."""
    pairs = []
    for i, p in enumerate(pred.itertuples()):
        for j, r in enumerate(ref.itertuples()):
            iou = interval_iou(p.t_start_ms, p.t_end_ms, r.t_start_ms, r.t_end_ms)
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


def block_metrics(pred, ref, min_iou=MATCH_IOU):
    """(summary dict, per-hit errors DataFrame) for one block."""
    m = match_reps(pred, ref, min_iou)
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
    blocks: iterable of (segmenter, block_id, pred_reps, ref_reps).
    Returns (per-block table, summary table with one row per segmenter).
    """
    rows, all_errs = [], []
    for seg, block, pred, ref in blocks:
        s, e = block_metrics(pred, ref, min_iou)
        rows.append({"segmenter": seg, "block_id": block, **s})
        all_errs.append(e.assign(segmenter=seg, block_id=block))
    per_block = pd.DataFrame(rows)
    errs = pd.concat(all_errs, ignore_index=True) if all_errs else pd.DataFrame()

    summary = []
    for seg, b in per_block.groupby("segmenter", sort=False):
        e = errs[errs["segmenter"] == seg]
        bound = pd.concat([e["err_start_ms"], e["err_end_ms"]]).dropna().to_numpy()
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
        yield "v0_fixed", v, segment_fixed(frames), truth
        yield "v1_rom", v, segment_reps(frames, rom, rest), truth


def real_blocks(frames, labels_dir, rest_ref, rom_ref, angle_col):
    """Both segmenters on each real clip that has a B9 labels file <block_id>_labels.csv."""
    from neuroflow.segmentation.label_tool import load_labels
    from neuroflow.segmentation.segment import segment_fixed, segment_reps
    for block, f in frames.groupby("block_id"):
        path = os.path.join(labels_dir, f"{block}_labels.csv")
        if not os.path.exists(path):
            print(f"  (no labels for {block}, skipped)")
            continue
        ref = load_labels(path)
        yield "v0_fixed", block, segment_fixed(f, angle_col=angle_col), ref
        yield "v1_rom", block, segment_reps(f, rom_ref, rest_ref, angle_col=angle_col), ref


def _print(per_block, summary):
    pd.set_option("display.width", 200)
    print(per_block.round(3).to_string(index=False))
    print()
    print(summary.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", help="folder with <block_id>_labels.csv from the B9 tool")
    p.add_argument("--frames", help="frames.parquet of the labelled clips")
    p.add_argument("--rest", type=float, help="rest_ref from a separate calibration block")
    p.add_argument("--rom", type=float, help="rom_ref from a separate calibration block")
    p.add_argument("--angle", default="theta_ht_deg")
    a = p.parse_args()

    # Hand check: IoU of [0, 10] and [5, 15] is 5 / 15.
    assert abs(interval_iou(0, 10, 5, 15) - 1 / 3) < 1e-12

    if a.labels:
        frames = pd.read_parquet(a.frames)
        print(f"S3 study on real clips ({a.angle}, rest {a.rest}, rom {a.rom}, IoU >= {MATCH_IOU})\n")
        _print(*study(real_blocks(frames, a.labels, a.rest, a.rom, a.angle)))
    else:
        print(f"S3 study on the mocks vs truth (IoU >= {MATCH_IOU}, tolerance +-{TOLERANCE_MS:.0f} ms)\n")
        _print(*study(mock_blocks()))


if __name__ == "__main__":
    main()
