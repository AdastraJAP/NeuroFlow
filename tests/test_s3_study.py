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
    pred = reps((10, 100), (0, 90))                             # both overlap ref; IoU .9 each...
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
