"""C8 tests: within-participant split vs LOSO (P1)."""
import numpy as np
import pandas as pd
import pytest

from fixtures.make_mock_reps import make_mock_reps
from neuroflow.scoring.folds import check_not_fitted_on, make_folds
from neuroflow.scoring.p1 import eval_reps, p1_table, run_within, within_split


@pytest.fixture(scope="module")
def reps():
    return make_mock_reps()


@pytest.fixture(scope="module")
def folds(reps):
    return make_folds(reps["subject_id"])


# ---- the split ----

def test_within_split_halves_each_subjects_s2_normal_test(reps, folds):
    tn = within_split(reps, folds, seed=0)
    picked = reps[reps["rep_id"].isin(tn)]
    assert (picked["session_id"] == "s2").all()
    assert (picked["block_id"] == "normal_test").all()
    dev = set(folds.loc[folds["split"] == "train", "subject_id"])
    assert set(picked["subject_id"]) == dev                 # every dev subject, no locked
    pool = reps[reps["subject_id"].isin(dev) & (reps["session_id"] == "s2")
                & (reps["block_id"] == "normal_test")]
    assert len(tn) == sum(len(g) // 2 for _, g in pool.groupby("subject_id"))
    assert within_split(reps, folds, seed=0) == tn          # fixed seed


def test_eval_set_is_test_half_plus_s2_conditions(reps, folds):
    tn = within_split(reps, folds)
    keep = eval_reps(reps, tn)
    rows = reps[reps["rep_id"].isin(keep)]
    assert set(rows.loc[rows["condition"] == "normal", "rep_id"]) == tn
    assert (rows.loc[rows["condition"] != "normal", "session_id"] == "s2").all()


def test_run_within_is_in_sample_and_holds_out_the_test_half(reps, folds):
    tn = within_split(reps, folds)
    seen = {}

    def fit(train):
        seen["train"] = set(train["rep_id"])
        return {"train_subjects": sorted(train["subject_id"].unique())}

    def score(model, r, fold_id=-1, allow_in_sample=False):
        check_not_fitted_on(model["train_subjects"], r, allow_in_sample)
        return pd.DataFrame({"rep_id": r["rep_id"].to_numpy(), "score_0_100": 50.0,
                             "abstain": False,
                             "ref_type": "in_sample" if allow_in_sample else "pooled"})

    out = run_within(reps, folds, fit, score, tn)
    assert not seen["train"] & tn                           # test half never in reference
    assert (out["ref_type"] == "in_sample").all()


# ---- the comparison can see a leak when there is one ----

def _synthetic(n_subjects=12, n_reps=20, tau=3.0, delta=0.8, seed=0):
    """
    One feature x. Big subject effect (tau), small condition effect (delta):
    across people the condition is hidden, within a person it is clear. A
    reference that contains the person's own normal reps can exploit that; LOSO
    cannot. That is the P1 mechanism.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        u = rng.normal(0, tau)
        for ses in ("s1", "s2"):
            for block, cond, shift in (("normal_test", "normal", 0.0), ("slow", "slow", delta)):
                for i in range(n_reps):
                    rows.append((f"p{s}_{ses}_{block}_{i}", f"p{s}", ses, block, cond, "2x",
                                 u + shift + rng.normal(0, 0.3)))
    reps = pd.DataFrame(rows, columns=["rep_id", "subject_id", "session_id", "block_id",
                                       "condition", "level", "x"])
    subj = sorted(reps["subject_id"].unique())
    folds = pd.DataFrame({"subject_id": subj, "fold_id": range(len(subj)), "split": "train"})
    return reps, folds


def _fit_nn(train):
    ref = train.loc[train["condition"] == "normal", "x"].to_numpy()
    return {"ref": np.sort(ref), "train_subjects": sorted(train["subject_id"].unique())}


def _score_nn(model, r, fold_id=-1, allow_in_sample=False):
    check_not_fitted_on(model["train_subjects"], r, allow_in_sample)
    d = np.abs(r["x"].to_numpy()[:, None] - model["ref"][None, :]).min(axis=1)
    return pd.DataFrame({"rep_id": r["rep_id"].to_numpy(), "score_0_100": -d,
                         "abstain": False, "fold_id": fold_id,
                         "ref_type": "in_sample" if allow_in_sample else "pooled"})


def test_p1_detects_a_leak_when_reference_holds_the_person():
    reps, folds = _synthetic()
    t = p1_table(reps, folds, _fit_nn, _score_nn, n_boot=300)
    row = t.iloc[0]
    assert row["gap"] > 0.1 and row["gap_excludes_zero"], row.to_dict()


def test_in_sample_scores_need_the_explicit_flag():
    from neuroflow.scoring.stats import paired_auroc_diff
    reps, folds = _synthetic(n_subjects=4, n_reps=5)
    tn = within_split(reps, folds)
    within = run_within(reps, folds, _fit_nn, _score_nn, tn)
    with pytest.raises(ValueError, match="in-sample"):
        paired_auroc_diff(within, within, reps, "slow", "2x", n_boot=10)
