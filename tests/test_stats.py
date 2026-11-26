"""C6 tests: subject bootstrap, paired differences, ICC/SEM/MDC."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from neuroflow.scoring.stats import (
    _boot_auroc, auroc, bootstrap_mean_ci, icc_2_1, paired_auroc_diff,
    percentile_ci, subject_draws, retest_agreement,
)


# ---- basics ----

def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    normal = rng.integers(0, 10, 40).astype(float)        # many ties on purpose
    impaired = rng.integers(0, 8, 30).astype(float)
    y = np.r_[np.zeros(40), np.ones(30)]
    assert auroc(normal, impaired) == pytest.approx(
        roc_auc_score(y, -np.r_[normal, impaired]))


def test_auroc_edge_cases():
    assert auroc([90.0], [10.0]) == 1.0
    assert auroc([100.0, 100.0], [100.0, 100.0]) == 0.5
    assert np.isnan(auroc([], [1.0]))


def test_draws_are_fixed_by_seed():
    np.testing.assert_array_equal(subject_draws(14, 50, seed=3), subject_draws(14, 50, seed=3))


def test_percentile_ci_drops_nan():
    lo, hi = percentile_ci([np.nan] + list(range(101)))
    assert (lo, hi) == pytest.approx((2.5, 97.5))


# ---- v0 ----

def test_mean_ci_resamples_subjects_not_reps():
    # 10 subjects, each with 50 identical reps: rep-level noise is zero, so the
    # whole interval width comes from the subjects. A rep bootstrap would claim
    # an interval far too narrow here.
    vals = np.repeat(np.arange(10.0), 50)
    subj = np.repeat([f"s{i}" for i in range(10)], 50)
    r = bootstrap_mean_ci(vals, subj, n_boot=2000, seed=0)
    assert r["estimate"] == pytest.approx(4.5)
    assert r["n_subjects"] == 10
    assert r["hi"] - r["lo"] > 2.0          # about 2 x 1.96 x 0.9 with 10 subjects


def test_subject_drawn_twice_counts_twice():
    normal = [np.array([1.0]), np.array([1.0])]
    impaired = [np.array([0.0]), np.array([2.0])]
    # draw subject 0 twice: both its pairs count, so AUROC = 1, not merged to 1 pair
    assert _boot_auroc(normal, impaired, np.array([[0, 0]]))[0] == 1.0
    assert _boot_auroc(normal, impaired, np.array([[0, 1]]))[0] == 0.5


# ---- "done when": CI covers a known synthetic effect about 95 % of the time ----

def _synthetic(rng, n_subjects, n_reps, delta, tau):
    """Subject effect u ~ N(0, tau^2) shared by all its reps; impaired shifted by -delta."""
    rows = []
    for s in range(n_subjects):
        u = rng.normal(0, tau)
        for cond, shift in (("normal", 0.0), ("slow", -delta)):
            for i in range(n_reps):
                rows.append((f"x{s}_{cond}_{i}", f"x{s}", cond, u + shift + rng.normal()))
    df = pd.DataFrame(rows, columns=["rep_id", "subject_id", "condition", "score_0_100"])
    reps = df.assign(session_id="s2", level="2x",
                     block_id=np.where(df["condition"] == "normal", "normal_test", "slow"))
    scores = df[["rep_id", "score_0_100"]].assign(abstain=False)
    return scores, reps


@pytest.mark.slow
def test_auroc_ci_coverage_on_known_effect():
    from neuroflow.scoring.stats import bootstrap_auroc
    delta, tau = 1.0, 0.5
    truth = norm.cdf(delta / np.sqrt(2 * (1 + tau ** 2)))   # pairs from random subjects
    rng = np.random.default_rng(2026)
    hits, n_sims = 0, 150
    for k in range(n_sims):
        scores, reps = _synthetic(rng, n_subjects=14, n_reps=15, delta=delta, tau=tau)
        r = bootstrap_auroc(scores, reps, "slow", "2x", n_boot=400, seed=k)
        hits += r["lo"] <= truth <= r["hi"]
    coverage = hits / n_sims
    print(f"coverage of the 95% subject-bootstrap CI: {coverage:.1%} (truth {truth:.3f})")
    # Percentile bootstrap with 14 clusters under-covers a little; 85-99 % is honest.
    assert 0.85 <= coverage <= 0.99


# ---- v2 ----

def test_paired_diff_is_zero_for_the_same_scorer():
    rng = np.random.default_rng(0)
    scores, reps = _synthetic(rng, n_subjects=10, n_reps=10, delta=1.0, tau=0.5)
    r = paired_auroc_diff(scores, scores, reps, "slow", "2x", n_boot=200)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0 and not r["excludes_zero"]


def test_paired_diff_uses_only_reps_both_scored():
    rng = np.random.default_rng(0)
    a, reps = _synthetic(rng, n_subjects=10, n_reps=10, delta=1.0, tau=0.5)
    b = a.copy()
    b.loc[b.index[:5], "abstain"] = True
    r = paired_auroc_diff(a, b, reps, "slow", "2x", n_boot=100)
    assert r["n_reps"] == len(a) - 5
    assert r["diff"] == 0                 # same scores on the shared reps


def test_paired_diff_detects_a_real_gap():
    rng = np.random.default_rng(0)
    good, reps = _synthetic(rng, n_subjects=14, n_reps=15, delta=2.0, tau=0.3)
    noisy = good.copy()
    noisy["score_0_100"] += rng.normal(0, 3, len(noisy))    # same reps, worse scorer
    r = paired_auroc_diff(good, noisy, reps, "slow", "2x", n_boot=500)
    assert r["diff"] > 0 and r["excludes_zero"]


def test_in_sample_scores_are_refused():
    from neuroflow.scoring.stats import bootstrap_auroc
    rng = np.random.default_rng(0)
    scores, reps = _synthetic(rng, n_subjects=5, n_reps=5, delta=1.0, tau=0.5)
    with pytest.raises(ValueError, match="in-sample"):
        bootstrap_auroc(scores.assign(ref_type="in_sample"), reps, "slow", "2x", n_boot=10)


# ---- ICC / SEM / MDC ----

def test_icc_matches_shrout_fleiss_1979():
    # Shrout & Fleiss (1979), Table 2: 6 targets x 4 judges; ICC(2,1) = 0.29
    x = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
         [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    assert icc_2_1(x) == pytest.approx(0.29, abs=0.005)


def test_icc_perfect_agreement_and_bias():
    x = np.c_[np.arange(10.0), np.arange(10.0)]
    assert icc_2_1(x) == pytest.approx(1.0)
    # a constant offset between sessions lowers ABSOLUTE agreement
    assert icc_2_1(np.c_[np.arange(10.0), np.arange(10.0) + 5]) < 0.8


def test_icc_refuses_missing_values():
    with pytest.raises(ValueError):
        icc_2_1([[1.0, np.nan], [2.0, 2.0]])


def test_retest_sem_and_mdc():
    subj = [f"s{i}" for i in range(8)]
    rows = [(f"{s}_{ses}", s, ses, "normal_test", 10.0 * i + (1.0 if ses == "s2" else 0.0))
            for i, s in enumerate(subj) for ses in ("s1", "s2")]
    reps = pd.DataFrame(rows, columns=["rep_id", "subject_id", "session_id", "block_id", "v"])
    scores = reps[["rep_id"]].assign(score_0_100=reps["v"], abstain=False)
    r = retest_agreement(scores, reps)
    assert r["n_subjects"] == 8
    assert r["icc"] > 0.99
    assert r["mdc95"] == pytest.approx(1.96 * np.sqrt(2) * r["sem"])
    # relative MDC is a function of ICC only, so it is comparable across scores
    assert r["mdc95_rel"] == pytest.approx(1.96 * np.sqrt(2 * (1 - r["icc"])))
    assert r["share_at_100"] == 0.0
