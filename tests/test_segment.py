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


# ---- angle input: theta_ht_deg (default) or theta_v_deg (A's V4 clips), issue #4 ----
import sys

sys.path.insert(0, FX)
from make_mock_frames import make_session  # noqa: E402

from neuroflow.segmentation.features import rep_features  # noqa: E402
from neuroflow.segmentation.segment import calibrate  # noqa: E402
from neuroflow.segmentation.templates import rep_matrix  # noqa: E402

CALIB_V = calibrate(make_session("normal", seed=7)[0], angle_col="theta_v_deg")


def _v4_like(frames):
    """Like A's seated V4 export: no hip landmarks, so theta_ht_deg is all NaN."""
    return frames.assign(theta_ht_deg=np.nan, trunk_lean_deg=np.nan)


@pytest.mark.parametrize("variant", ["normal", "slow", "capped"])
def test_theta_v_segments_like_theta_ht(variant):
    frames, truth, _, _ = _load(variant)
    rest, rom = CALIB_V
    reps = segment_reps(_v4_like(frames), rom, rest, angle_col="theta_v_deg")
    ok, n, _ = validate(frames, truth, reps=reps)
    assert ok and n == len(truth)
    assert (reps["angle_col"] == "theta_v_deg").all()


def test_default_angle_is_recorded():
    frames, _, rest, rom = _load("normal")
    assert (segment_reps(frames, rom, rest)["angle_col"] == "theta_ht_deg").all()
    assert (segment_fixed(frames)["angle_col"] == "theta_ht_deg").all()


def test_missing_angle_fails_loudly_not_zero_reps():
    frames, _, rest, rom = _load("normal")
    with pytest.raises(ValueError, match="theta_ht_deg has no values"):
        segment_reps(_v4_like(frames), rom, rest)
    with pytest.raises(ValueError, match="theta_ht_deg has no values"):
        calibrate(_v4_like(frames))
    with pytest.raises(ValueError, match="angle_col must be one of"):
        segment_fixed(frames, angle_col="elbow_deg")


def test_downstream_follows_the_segmented_angle():
    frames, _, _, _ = _load("normal")
    rest, rom = CALIB_V
    v4 = _v4_like(frames)
    reps = segment_reps(v4, rom, rest, angle_col="theta_v_deg")
    feats = rep_features(v4, reps, rest, rom)              # picks theta_v_deg from reps
    assert not feats[["rom_peak_deg", "sparc"]].isna().any().any()
    assert feats["max_trunk_lean_deg"].isna().all()        # no hips -> no trunk lean
    R, kept = rep_matrix(v4, reps, rest, rom)
    assert R.shape[0] == len(reps)
    with pytest.raises(ValueError, match="segmented on theta_v_deg"):
        rep_features(v4, reps, rest, rom, angle_col="theta_ht_deg")
    with pytest.raises(ValueError, match="segmented on theta_v_deg"):
        rep_matrix(v4, reps, rest, rom, angle_col="theta_ht_deg")


def test_theta_v_reads_trunk_lean_as_extra_range():
    """Known limit of theta_v: on the leaning mock it adds the ~12 deg lean to the range."""
    frames, _, rest, rom = _load("leaning")
    reps = segment_reps(frames, rom, rest)
    ht = rep_features(frames, reps, rest, rom)["rom_peak_deg"].mean()
    rest_v, rom_v = CALIB_V
    reps_v = segment_reps(frames, rom_v, rest_v, angle_col="theta_v_deg")
    v = rep_features(frames, reps_v, rest_v, rom_v)["rom_peak_deg"].mean()
    assert v - ht == pytest.approx(12.0, abs=1.5)
