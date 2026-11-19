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


# ---- B3: ROM-normalised segmenter ----
import numpy as np

from neuroflow.segmentation.segment import calibration_refs, segment_fixed, segment_reps

CALIB = calibration_refs()  # from separate calibration blocks, never the block under test


def _load(variant):
    frames = pd.read_parquet(os.path.join(FX, f"mock_frames_{variant}.parquet"))
    truth = pd.read_csv(os.path.join(FX, f"mock_frames_{variant}_truth.csv"))
    rest, rom = CALIB[frames["subject_id"].iloc[0]]
    return frames, truth, rest, rom


@pytest.mark.parametrize("variant", ["normal", "slow", "capped", "leaning", "low_rom"])
def test_b3_finds_every_rep(variant):
    frames, truth, rest, rom = _load(variant)
    assert len(segment_reps(frames, rom, rest)) == len(truth)


@pytest.mark.parametrize("variant", ["normal", "slow", "capped", "leaning"])
def test_b3_boundaries_match_truth(variant):
    # low_rom is excluded on purpose: 5% of a ~15 deg range is below the noise
    # (documented known limit in tasks.md); only its rep count is tested.
    frames, truth, rest, rom = _load(variant)
    ok, n, err = validate(frames, truth, reps=segment_reps(frames, rom, rest))
    assert ok, f"worst boundary error > {TOLERANCE_MS} ms"


def test_v0_misses_low_rom_user():
    # Why B3 exists: fixed 30 deg thresholds never fire for a user who peaks ~25 deg.
    frames, _, _, _ = _load("low_rom")
    assert len(segment_fixed(frames)) == 0


def test_b3_no_false_reps_at_rest():
    frames, _, _, _ = _load("normal")
    frames = frames.copy()
    frames["theta_ht_deg"] = 10 + np.random.default_rng(0).normal(0, 1, len(frames))
    for rest, rom in CALIB.values():
        assert len(segment_reps(frames, rom, rest)) == 0
