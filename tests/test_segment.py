"""Golden-file test for B2: every rep found, start and end within +-67 ms of truth."""
import os

import pandas as pd
import pytest

from neuroflow.segmentation.segment import TOLERANCE_MS, validate

FX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


@pytest.mark.parametrize("variant", ["normal", "slow", "capped", "leaning"])
def test_b2_matches_truth(variant):
    frames = pd.read_parquet(os.path.join(FX, f"mock_frames_{variant}.parquet"))
    truth = pd.read_csv(os.path.join(FX, f"mock_frames_{variant}_truth.csv"))
    ok, n, err = validate(frames, truth)
    assert n == len(truth), f"found {n} reps, expected {len(truth)}"
    worst = max(err["err_start_ms"].abs().max(), err["err_end_ms"].abs().max())
    assert ok, f"worst boundary error {worst:.1f} ms > {TOLERANCE_MS} ms"
