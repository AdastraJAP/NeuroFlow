"""B8 tests: IoU, one-to-one matching, block metrics and the pooled summary."""
import numpy as np
import pandas as pd
import pytest

from neuroflow.segmentation.s3_study import (
    block_metrics, interval_iou, match_reps, mock_blocks, study)


def reps(*windows):
    return pd.DataFrame([{"t_start_ms": s, "t_peak_ms": (s + e) / 2, "t_end_ms": e} for s, e in windows])


def test_iou_hand_examples():
    assert interval_iou(0, 10, 5, 15) == pytest.approx(1 / 3)
    assert interval_iou(0, 10, 0, 10) == 1.0
    assert interval_iou(0, 10, 20, 30) == 0.0
    assert np.isnan(interval_iou(np.nan, 10, 0, 10))
    assert np.isnan(interval_iou(10, 10, 0, 10))                # empty window


def test_matching_is_one_to_one_best_first():
    ref = reps((0, 100))
    pred = reps((10, 100), (0, 90))                             # both overlap ref (IoU 0.9 each)
    m = match_reps(pred, ref)
    assert len(m) == 1                                          # ref used once
    pred = reps((0, 100), (300, 400))
    ref = reps((5, 100), (290, 400), (600, 700))
    assert [(i, j) for i, j, _ in match_reps(pred, ref)] == [(0, 0), (1, 1)]


def test_block_metrics_counts_and_errors():
    ref = reps((0, 1000), (2000, 3000), (4000, 5000))
    pred = reps((30, 980), (2000, 3050), (6000, 7000))          # 2 hits, 1 false alarm, 1 miss
    s, e = block_metrics(pred, ref)
    assert (s["n_pred"], s["n_ref"], s["count_err"], s["hits"]) == (3, 3, 0, 2)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert sorted(e["err_start_ms"]) == [0, 30] and sorted(e["err_end_ms"]) == [-20, 50]


def test_unknown_boundary_is_never_a_hit():
    s, _ = block_metrics(reps((np.nan, 1000)), reps((0, 1000)))
    assert s["hits"] == 0 and s["recall"] == 0


def test_empty_blocks():
    s, _ = block_metrics(reps(), reps((0, 1000)))
    assert s["count_err"] == -1 and s["recall"] == 0 and np.isnan(s["precision"])


def test_summary_rates():
    ref = reps((0, 1000))
    blocks = [("a", "b1", reps((0, 1000)), ref),                # exact
              ("a", "b2", reps((0, 1000), (2000, 3000)), ref),  # off by one
              ("a", "b3", reps(), ref)]                         # off by one (missed)
    per_block, summary = study(blocks)
    r = summary.iloc[0]
    assert r["count_mae"] == pytest.approx(2 / 3)
    assert r["exact_count_rate"] == pytest.approx(1 / 3) and r["off_by_one_rate"] == pytest.approx(2 / 3)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["within_tol_rate"] == 1.0


def test_mocks_reproduce_b3_results():
    """On the mocks, v1 finds every rep everywhere; v0 misses all of low_rom (B3's finding)."""
    per_block, summary = study(mock_blocks())
    v1 = summary.set_index("segmenter").loc["v1_rom"]
    assert v1["count_mae"] == 0 and v1["f1"] == 1.0
    low = per_block.set_index(["segmenter", "block_id"]).loc[("v0_fixed", "low_rom")]
    assert low["n_pred"] == 0 and low["recall"] == 0


# ---- real clips: labels must match the clip and its clock (Adastra, PR #3 review) ----
import os  # noqa: E402

from neuroflow.segmentation.label_tool import Labels, save_labels  # noqa: E402
from neuroflow.segmentation.s3_study import check_labels, real_blocks  # noqa: E402
from neuroflow.segmentation.segment import calibration_refs  # noqa: E402

FX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _clip(variant):
    frames = pd.read_parquet(os.path.join(FX, f"mock_frames_{variant}.parquet"))
    truth = pd.read_csv(os.path.join(FX, f"mock_frames_{variant}_truth.csv"))
    return frames, truth


def _write_labels(folder, frames, truth, **override):
    """B9-style labels at the truth times; override changes the identity/clock fields."""
    lab = Labels()
    for _, r in truth.iterrows():
        for c in ("t_start_ms", "t_peak_ms", "t_end_ms"):
            lab.mark(r[c])
    first = frames.iloc[0]
    meta = dict(subject_id=first["subject_id"], session_id=first["session_id"],
                block_id=first["block_id"], labeler="test", time_source="t_capture_ms")
    meta.update(override)
    save_labels(lab.to_frame(**meta), os.path.join(folder, f"{first['block_id']}_labels.csv"))


def _run_real(frames, folder, **kw):
    rest, rom = calibration_refs()[frames["subject_id"].iloc[0]]
    return study(real_blocks(frames, str(folder), rest, rom, "theta_ht_deg", **kw))


def test_real_blocks_with_matching_labels(tmp_path):
    frames, truth = _clip("normal")
    _write_labels(tmp_path, frames, truth)
    per_block, summary = _run_real(frames, tmp_path)
    assert summary.set_index("segmenter").loc["v1_rom", "f1"] == 1.0


@pytest.mark.parametrize("field, value, match", [
    ("time_source", "video_fps", "time_source must be t_capture_ms"),
    ("block_id", "slow", "block_id"),
    ("subject_id", "mock99", "subject_id"),
    ("session_id", "s9", "session_id"),
])
def test_mismatched_labels_are_rejected(tmp_path, field, value, match):
    frames, truth = _clip("normal")
    _write_labels(tmp_path, frames, truth, **{field: value})
    with pytest.raises(ValueError, match=match):
        _run_real(frames, tmp_path)


def test_missing_label_files_are_reported(tmp_path, capsys):
    a, ta = _clip("normal")
    b, _ = _clip("slow")
    both = pd.concat([a, b], ignore_index=True)
    _write_labels(tmp_path, a, ta)                      # no labels for "slow"
    with pytest.raises(FileNotFoundError, match=r"1 of 2 expected label files are missing.*slow_labels"):
        _run_real(both, tmp_path)
    per_block, _ = _run_real(both, tmp_path, allow_missing=True)
    assert set(per_block["block_id"]) == {"normal"}
    assert "slow_labels.csv" in capsys.readouterr().out


def test_check_labels_accepts_an_empty_labels_file():
    frames, _ = _clip("normal")
    check_labels(Labels().to_frame(), frames, "empty.csv")


@pytest.mark.parametrize("shift_ms, point", [(-1e6, "t_start_ms"), (1e6, "t_end_ms")])
def test_labels_outside_the_clip_are_rejected(tmp_path, shift_ms, point):
    frames, truth = _clip("normal")
    moved = truth.copy()
    moved.loc[0 if shift_ms < 0 else len(moved) - 1, point] += shift_ms
    _write_labels(tmp_path, frames, moved)
    with pytest.raises(ValueError, match="outside the clip"):
        _run_real(frames, tmp_path)


def test_labels_at_the_clip_edges_are_accepted(tmp_path):
    frames, truth = _clip("normal")
    edge = truth.copy()
    edge.loc[len(edge) - 1, "t_end_ms"] = frames["t_capture_ms"].max()   # like T_02's end
    _write_labels(tmp_path, frames, edge)
    _run_real(frames, tmp_path)
