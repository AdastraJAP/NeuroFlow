"""B5 tests: rep templates."""
import os

import numpy as np
import pandas as pd
import pytest

from neuroflow.segmentation.dtw import dtw_distance
from neuroflow.segmentation.segment import calibration_refs, segment_reps
from neuroflow.segmentation.dtw import dtw_path
from neuroflow.segmentation.templates import (
    TEMPLATE_LEN, dba_template, expected_shape, mean_template, rep_matrix, resample_rep,
    spread, timing_varied_reps)

FX = os.path.join(os.path.dirname(__file__), "..", "fixtures")
CALIB = calibration_refs()


def _matrix(variant):
    f = pd.read_parquet(os.path.join(FX, f"mock_frames_{variant}.parquet"))
    tr = pd.read_csv(os.path.join(FX, f"mock_frames_{variant}_truth.csv"))
    rest, rom = CALIB[f["subject_id"].iloc[0]]
    R, _ = rep_matrix(f, segment_reps(f, rom, rest), rest, rom)
    return R, tr, rest, rom


# ---- step 1: linear-stretch mean ----

def test_resample_is_exact_on_a_line():
    t = np.arange(0, 1000, 33.0)
    x = resample_rep(t, 2 * t, 100.0, 700.0, n=7)
    assert np.allclose(x, 2 * np.linspace(100, 700, 7))


def test_resample_rejects_bad_reps():
    t = np.arange(0, 1000, 33.0)
    assert resample_rep(t, t, np.nan, 500.0) is None          # unknown boundary
    assert resample_rep(t, t, 100.0, 150.0) is None           # too few frames
    assert resample_rep(t, np.full(len(t), np.nan), 0, 900) is None


def test_mean_template_recovers_true_shape():
    R, tr, rest, rom = _matrix("normal")
    T = mean_template(R)
    E = expected_shape(tr, rest, rom)
    assert R.shape == (10, TEMPLATE_LEN)
    assert np.sqrt(np.mean((T - E) ** 2)) < 0.02          # < ~1.6 deg on an 80 deg range
    assert np.corrcoef(T, E)[0, 1] > 0.99
    assert 28 <= int(np.argmax(T)) <= 31                  # symmetric rep peaks mid-way
    assert T[0] < 0.1 and T[-1] < 0.1                     # ends near the 5% level


def test_stretching_removes_overall_tempo():
    T_norm = mean_template(_matrix("normal")[0])
    T_slow = mean_template(_matrix("slow")[0])
    assert dtw_distance(T_norm, T_slow) < 0.1


def test_capped_template_keeps_smaller_range():
    assert mean_template(_matrix("capped")[0]).max() < 0.6


def test_mean_template_rejects_empty():
    with pytest.raises(ValueError):
        mean_template(np.empty((0, TEMPLATE_LEN)))


# ---- step 2: DBA ----

def test_dtw_path_matches_hand_example():
    d, path = dtw_path([0, 1, 2, 1, 0], [0, 0, 1, 2, 1], None)
    assert d == 1.0
    assert path == [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 4)]


def test_dba_objective_never_increases():
    for R in (_matrix("normal")[0], timing_varied_reps()):
        _, hist = dba_template(R)
        assert all(h1 >= h2 - 1e-9 for h1, h2 in zip(hist, hist[1:]))


def test_dba_of_identical_reps_is_that_rep():
    x = timing_varied_reps(k=1, noise=0.0)[0]
    T, _ = dba_template(np.tile(x, (5, 1)))
    assert np.allclose(T, x)


@pytest.mark.parametrize("which", ["normal", "synth"])
def test_dba_beats_mean_on_distance_to_members(which):
    R = _matrix("normal")[0] if which == "normal" else timing_varied_reps()
    T_dba, _ = dba_template(R)
    assert spread(R, T_dba).mean() < spread(R, mean_template(R)).mean()


def test_dba_keeps_peak_when_timing_varies():
    # Reps peak anywhere in 35-65 % of the rep: the plain mean blurs the peak, DBA does not.
    R = timing_varied_reps()
    assert mean_template(R).max() < 0.95
    assert abs(dba_template(R)[0].max() - 1.0) < 0.02


def test_dba_stays_close_to_true_shape_on_mock():
    # Known behaviour: with identical rep timing DBA fits some noise (about 1 deg here),
    # so it is less exact than the mean on the mocks -- guard that it stays small.
    R, tr, rest, rom = _matrix("normal")
    T, _ = dba_template(R)
    assert np.sqrt(np.mean((T - expected_shape(tr, rest, rom)) ** 2)) < 0.02
