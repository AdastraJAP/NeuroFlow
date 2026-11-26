"""C4 tests: sub-scores and feedback codes."""
import json

import numpy as np
import pandas as pd
import pytest

from fixtures.make_mock_reps import make_mock_reps
from neuroflow.scoring.contract import FEEDBACK_CODES, SUBSCORE_KEYS
from neuroflow.scoring.folds import iter_folds, make_folds
from neuroflow.scoring.scorer import fit_scorer, score_reps
from neuroflow.scoring.subscores import (
    SUBSCORE_SPEC, _deviation, _robust_scale, feedback_codes, fit_subscores,
    subscore_table,
)

EXPECTED_CODE = {
    "slow": "tempo", "band": "tempo", "trunk_lean": "trunk",
    "elbow_bend": "elbow", "rom_cap": "rom",
}


@pytest.fixture(scope="module")
def reps():
    return make_mock_reps()


@pytest.fixture(scope="module")
def loso_s2(reps):
    """Every development subject scored by a model that never saw them; session 2 only."""
    folds = make_folds(reps["subject_id"])
    parts = [score_reps(fit_scorer(train), test, fold_id=fold_id)
             for fold_id, train, test in iter_folds(reps, folds)]
    out = pd.concat(parts, ignore_index=True).merge(
        reps[["rep_id", "session_id", "condition", "level"]], on="rep_id")
    return out[out["session_id"] == "s2"]


# ---- "done when": a lean-heavy rep returns feedback code "trunk" ----

@pytest.mark.parametrize("condition", sorted(EXPECTED_CODE))
def test_condition_gets_its_feedback_code(loso_s2, condition):
    codes = loso_s2.loc[loso_s2["condition"] == condition, "feedback_code"].dropna()
    assert codes.value_counts().idxmax() == EXPECTED_CODE[condition]


def test_lean_heavy_rep_says_trunk(loso_s2):
    lean = loso_s2[(loso_s2["condition"] == "trunk_lean") & (loso_s2["level"] == "large")]
    assert (lean["feedback_code"] == "trunk").mean() > 0.9


# ---- false feedback on normal reps (measured, not assumed) ----

def test_false_feedback_rate_on_heldout_normals(loso_s2):
    normal = loso_s2[(loso_s2["condition"] == "normal") & ~loso_s2["abstain"]]
    rate = normal["feedback_code"].notna().mean()
    print(f"false-feedback rate on held-out normal reps: {rate:.1%}")
    # Measured 6.8% on the mock (seed 0): above the naive 5% because held-out
    # subjects score lower than training subjects (C3: median 40.5, not 50).
    assert rate < 0.10


# ---- contract ----

def test_codes_are_allowed_and_hike_never_speaks(loso_s2):
    codes = set(loso_s2["feedback_code"].dropna())
    assert codes <= set(FEEDBACK_CODES)
    assert "hike" not in codes


def test_subscores_json_has_every_key(loso_s2):
    ok = loso_s2[~loso_s2["abstain"]]
    keys = set(json.loads(ok["subscores"].iloc[0]))
    # the mock has no shoulder-tilt column, so "tilt" is left out of the model
    assert keys == set(SUBSCORE_KEYS) - {"tilt"}


def test_abstained_reps_get_no_feedback(loso_s2):
    ab = loso_s2[loso_s2["abstain"]]
    assert len(ab) > 0
    assert ab["feedback_code"].isna().all()


# ---- fitting rules ----

def test_fit_uses_normal_reps_only(reps):
    train = reps[reps["subject_id"] != "sub01"]
    only_normal = train[(train["condition"] == "normal") | (train["block_id"] == "normal_calib")]
    a, b = fit_subscores(train), fit_subscores(only_normal)
    assert set(a) == set(b)
    for name in a:
        assert a[name]["median"] == b[name]["median"]
        np.testing.assert_array_equal(a[name]["ref"], b[name]["ref"])


def test_training_normals_are_roughly_uniform(reps):
    model = fit_subscores(reps)
    normal = reps[reps["condition"] == "normal"]
    subs = subscore_table(model, normal)
    for name in ("shape", "trunk", "elbow"):          # continuous one-sided features
        assert 40 < np.nanmedian(subs[name]) < 60, name


# ---- helpers ----

def test_direction_flips():
    x = np.array([1.0, 3.0])
    assert list(_deviation(x, 2.0, 1.0, "high")) == [-1.0, 1.0]
    assert list(_deviation(x, 2.0, 1.0, "low")) == [1.0, -1.0]
    assert list(_deviation(x, 2.0, 1.0, "two")) == [1.0, 1.0]


def test_zero_mad_falls_back():
    assert _robust_scale(np.array([5.0, 5.0, 5.0])) == 1.0           # constant
    x = np.array([0.0, 0.0, 0.0, 0.0, 10.0])                          # MAD 0, SD > 0
    assert _robust_scale(x) == pytest.approx(np.std(x, ddof=1))


def test_ties_go_to_largest_deviation():
    codes = list(FEEDBACK_CODES)
    subs = pd.DataFrame([[0.0] * len(codes)], columns=codes)
    dev = pd.DataFrame([[1.0] * len(codes)], columns=codes)
    dev.loc[0, "elbow"] = 9.0
    assert feedback_codes([0.0], subs, dev).iloc[0] == "elbow"


def test_no_feedback_when_global_score_is_fine():
    codes = list(FEEDBACK_CODES)
    subs = pd.DataFrame([[0.0] * len(codes)], columns=codes)
    dev = pd.DataFrame([[5.0] * len(codes)], columns=codes)
    assert feedback_codes([50.0], subs, dev).iloc[0] is None
