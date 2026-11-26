"""C3 tests: scorer v1 (Ledoit-Wolf Mahalanobis + pooled ECDF)."""
import json

import numpy as np
import pandas as pd
import pytest

from fixtures.make_mock_reps import make_mock_reps
from neuroflow.scoring.contract import SCORE_COLUMNS, SUBSCORE_KEYS
from neuroflow.scoring.folds import iter_folds, make_folds
from neuroflow.scoring.scorer import feature_matrix, fit_scorer, score_reps


@pytest.fixture(scope="module")
def reps():
    return make_mock_reps()


@pytest.fixture(scope="module")
def first_fold(reps):
    folds = make_folds(reps["subject_id"])
    fold_id, train, test = next(iter(iter_folds(reps, folds)))
    return fold_id, train, test, fit_scorer(train)


@pytest.fixture(scope="module")
def loso_scores(reps):
    """Every development subject scored once, by a model that never saw them."""
    folds = make_folds(reps["subject_id"])
    parts = [score_reps(fit_scorer(train), test, fold_id=fold_id)
             for fold_id, train, test in iter_folds(reps, folds)]
    scores = pd.concat(parts, ignore_index=True)
    return scores.merge(reps[["rep_id", "session_id", "condition", "level"]], on="rep_id")


# ---- "done when" 1: a shifted mock group scores lower than normal ----

@pytest.mark.parametrize("condition", ["slow", "band", "trunk_lean", "elbow_bend"])
def test_shifted_group_scores_lower(loso_scores, condition):
    s2 = loso_scores[loso_scores["session_id"] == "s2"]
    normal = s2.loc[s2["condition"] == "normal", "score_0_100"].median()
    shifted = s2.loc[s2["condition"] == condition, "score_0_100"].median()
    assert shifted < normal - 20, (condition, shifted, normal)


def test_slow_reps_no_longer_saturate(loso_scores):
    # The RehabHub flaw (C2): ~75 % of slow reps score exactly 100.
    slow = loso_scores.loc[loso_scores["condition"] == "slow", "score_0_100"].dropna()
    assert (slow == 100).mean() < 0.02


# ---- "done when" 2: ECDF of the training normals is about uniform ----

def test_training_normals_ecdf_is_uniform(first_fold):
    _, train, _, model = first_fold
    normal = train[train["condition"] == "normal"]
    s = score_reps(model, normal, allow_in_sample=True)["score_0_100"].dropna().to_numpy()
    assert s.mean() == pytest.approx(50, abs=2)
    # each tenth of 0-100 holds about a tenth of the reps
    counts, _ = np.histogram(s, bins=10, range=(0, 100))
    assert np.all(np.abs(counts / len(s) - 0.1) < 0.02), counts


# ---- distance kept in subscores ----

def test_subscores_hold_distance(first_fold):
    fold_id, _, test, model = first_fold
    out = score_reps(model, test, fold_id=fold_id)
    ok = out[~out["abstain"]]
    d = np.array([json.loads(x)["d"] for x in ok["subscores"]])
    assert set(json.loads(ok["subscores"].iloc[0])) <= set(SUBSCORE_KEYS)
    assert np.all(d >= 0)
    # a larger distance never gives a higher score
    order = np.argsort(d)
    assert np.all(np.diff(ok["score_0_100"].to_numpy()[order]) <= 1e-9)


def test_abstained_reps_have_no_subscores(first_fold):
    fold_id, _, test, model = first_fold
    out = score_reps(model, test, fold_id=fold_id)
    ab = out[out["abstain"]]
    assert len(ab) > 0                      # the mock has some NaN starts
    assert ab["subscores"].isna().all()
    assert ab["score_0_100"].isna().all()
    # the mock's unknown starts abstain as censored, with the reason named
    assert (ab["abstain_reason"] == "censored_start").all()


# ---- contract and leak guards ----

def test_output_follows_contract(first_fold):
    fold_id, _, test, model = first_fold
    out = score_reps(model, test, fold_id=fold_id)
    assert list(out.columns) == SCORE_COLUMNS
    assert len(out) == len(test)
    s = out["score_0_100"].dropna()
    assert s.between(0, 100).all()


def test_refuses_to_score_training_subjects(first_fold):
    fold_id, train, _, model = first_fold
    with pytest.raises(ValueError, match="fitted on"):
        score_reps(model, train, fold_id=fold_id)


def test_fit_uses_normal_train_reps_only(first_fold):
    _, train, _, model = first_fold
    # changing impaired training reps must not change the model
    other = train.copy()
    other.loc[other["condition"] != "normal", "dtw_dist"] *= 10
    m2 = fit_scorer(other)
    assert np.allclose(model["d_ref"], m2["d_ref"])


def test_missing_calibration_fails_loudly(reps):
    no_calib = reps[reps["block_id"] != "normal_calib"]
    with pytest.raises(ValueError, match="normal_calib"):
        feature_matrix(no_calib)


def test_all_nan_calibration_fails_loudly(reps):
    bad = reps.copy()
    is_calib = (bad["subject_id"] == "sub01") & (bad["block_id"] == "normal_calib")
    bad.loc[is_calib, "duration_s"] = np.nan
    with pytest.raises(ValueError, match="sub01"):
        feature_matrix(bad)


# ---- in-sample scoring needs an explicit flag (Melchior, PR #8 review, point 3) ----

def test_locked_fold_id_does_not_switch_off_leak_check(first_fold):
    _, train, _, model = first_fold
    with pytest.raises(ValueError, match="fitted on"):
        score_reps(model, train, fold_id=-1)          # -1 is LOCKED_FOLD_ID, not a bypass


def test_in_sample_rows_are_marked(first_fold):
    _, train, _, model = first_fold
    out = score_reps(model, train.head(20), allow_in_sample=True)
    assert (out["ref_type"] == "in_sample").all()
