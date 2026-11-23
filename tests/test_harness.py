"""C5 tests: LOSO harness, per-condition AUROC and the leak test."""
import numpy as np
import pandas as pd
import pytest

from fixtures.make_mock_reps import make_mock_reps
from neuroflow.scoring.baseline import fit_baseline, score_baseline
from neuroflow.scoring.folds import make_folds
from neuroflow.scoring.harness import (
    AUROC_COLUMNS, SHUFFLE_TOLERANCE, _auroc, auroc_by_condition,
    fit_nearest_normal, run_loso, score_nearest_normal, shuffle_within_subject,
    shuffled_label_check,
)
from neuroflow.scoring.scorer import fit_scorer, score_reps


@pytest.fixture(scope="module")
def reps():
    return make_mock_reps()


@pytest.fixture(scope="module")
def folds(reps):
    return make_folds(reps["subject_id"])


@pytest.fixture(scope="module")
def c3_table(reps, folds):
    return auroc_by_condition(run_loso(reps, folds, fit_scorer, score_reps), reps)


@pytest.fixture(scope="module")
def base_table(reps, folds):
    return auroc_by_condition(run_loso(reps, folds, fit_baseline, score_baseline), reps)


# ---- run_loso ----

def test_every_dev_rep_scored_once(reps, folds):
    scores = run_loso(reps, folds, fit_scorer, score_reps)
    dev = folds.loc[folds["split"] == "train", "subject_id"]
    expected = reps[reps["subject_id"].isin(dev)]
    assert scores["rep_id"].is_unique
    assert set(scores["rep_id"]) == set(expected["rep_id"])


def test_locked_subjects_never_scored(reps, folds):
    scores = run_loso(reps, folds, fit_baseline, score_baseline)
    locked = set(folds.loc[folds["split"] == "locked_test", "subject_id"])
    scored = set(reps.loc[reps["rep_id"].isin(scores["rep_id"]), "subject_id"])
    assert not scored & locked


# ---- auroc_by_condition ----

def test_table_shape_and_control_flag(c3_table):
    assert list(c3_table.columns) == AUROC_COLUMNS
    assert "normal" not in set(c3_table["condition"])
    assert c3_table.loc[c3_table["condition"] == "rom_cap", "positive_control"].all()
    assert not c3_table.loc[c3_table["condition"] != "rom_cap", "positive_control"].any()


def test_normal_side_is_normal_test_block_s2(reps, folds, c3_table):
    scores = run_loso(reps, folds, fit_scorer, score_reps)
    ok = set(scores.loc[~scores["abstain"], "rep_id"])
    want = reps[(reps["session_id"] == "s2") & (reps["block_id"] == "normal_test")
                & reps["rep_id"].isin(ok)]
    assert (c3_table["n_normal"] == len(want)).all()     # no calib, no s1


def test_coverage_reported(c3_table):
    assert ((c3_table["coverage"] > 0.9) & (c3_table["coverage"] <= 1.0)).all()


def test_c3_catches_slow_reps_baseline_does_not(c3_table, base_table):
    for lvl in ("2x", "3x"):
        c3 = c3_table.query("condition == 'slow' and level == @lvl")["auroc"].item()
        b = base_table.query("condition == 'slow' and level == @lvl")["auroc"].item()
        assert c3 > 0.9 and b < 0.65


def test_auroc_ties_count_half():
    assert _auroc(np.array([100.0, 100.0]), np.array([100.0, 100.0])) == 0.5
    assert _auroc(np.array([90.0]), np.array([10.0])) == 1.0


# ---- leak test ----

def test_shuffle_keeps_pairs_and_calibration(reps):
    sh = shuffle_within_subject(reps, seed=0)
    calib = reps["block_id"] == "normal_calib"
    pd.testing.assert_frame_equal(sh[calib], reps[calib])
    pairs = lambda d: sorted(zip(d["subject_id"], d["condition"], d["level"]))
    assert pairs(sh) == pairs(reps)          # same pairs per subject, new places


def test_c3_passes_shuffled_label_check(reps, folds):
    a = shuffled_label_check(reps, folds, fit_scorer, score_reps, n_shuffles=5, seed=0)
    assert abs(a.mean() - 0.5) <= SHUFFLE_TOLERANCE


def test_positive_control_fails_when_it_leaks(reps, folds):
    # A memorising scorer fitted on the test subject must be caught: if this
    # stays near 0.5, the leak test is not sensitive and proves nothing.
    a = shuffled_label_check(reps, folds, fit_nearest_normal, score_nearest_normal,
                             n_shuffles=1, seed=0, leak=True)
    assert a.mean() > 0.5 + 4 * SHUFFLE_TOLERANCE


def test_positive_control_passes_without_leak(reps, folds):
    a = shuffled_label_check(reps, folds, fit_nearest_normal, score_nearest_normal,
                             n_shuffles=3, seed=0)
    assert abs(a.mean() - 0.5) <= SHUFFLE_TOLERANCE


def test_auroc_table_refuses_in_sample_rows(reps, folds):
    scores = run_loso(reps, folds, fit_baseline, score_baseline, leak=True)
    assert (scores["ref_type"] == "in_sample").all()
    with pytest.raises(ValueError, match="in-sample"):
        auroc_by_condition(scores, reps)
