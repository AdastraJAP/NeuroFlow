"""
C2: RehabHub baseline scorer (owner: Member C).

Reproduces the scoring of Labchurat et al. 2026 (RehabHub), as reported in the
research basis:
    Z = (d - mu) / sigma        d = the rep's DTW distance to the template
                                mu, sigma from the normal reps of the training set
    score = 100 / max(Z, 1)

Known weakness this baseline must reproduce (it is what P1/P2 measure):
every rep with Z below 1 (closer than one SD above the normal mean) scores 100,
and DTW ignores speed, so slow but full-range reps also score 100.

Rule: fit on training subjects only. fit_baseline() uses only the reps you pass
it, and only their normal reps; the caller (the LOSO harness, C5) decides which
subjects those are.

Run (from the repo root):
    python -m neuroflow.scoring.baseline
"""

import numpy as np
import pandas as pd

from neuroflow.scoring.contract import SCHEMA_VERSION, SCORE_COLUMNS

SCORER_ID = "rehabhub_z"
MAPPING = "100/max(Z,1)"
NORMAL = "normal"


def fit_baseline(train_reps):
    """mu and sigma of the DTW distance over the normal training reps."""
    d = train_reps.loc[train_reps["condition"] == NORMAL, "dtw_dist"].dropna()
    if len(d) < 2:
        raise ValueError("need at least 2 normal reps with a DTW distance")
    sigma = float(d.std(ddof=1))
    if sigma == 0:
        raise ValueError("all normal DTW distances are equal; sigma is 0")
    return {
        "scorer_id": SCORER_ID,
        "mu": float(d.mean()),
        "sigma": sigma,
        "n_normal": int(len(d)),
        "train_subjects": sorted(train_reps["subject_id"].unique()),
    }


def z_score(model, dtw_dist):
    return (np.asarray(dtw_dist, dtype=float) - model["mu"]) / model["sigma"]


def rehabhub_score(z):
    """100 / max(Z, 1). NaN in, NaN out."""
    z = np.asarray(z, dtype=float)
    return 100.0 / np.maximum(z, 1.0)


def score_baseline(model, reps, fold_id=-1, ref_type="pooled_mean"):
    """Score reps with a fitted model. Returns a scores table (SCORE_COLUMNS)."""
    overlap = set(model["train_subjects"]) & set(reps["subject_id"])
    if overlap and fold_id >= 0:
        raise ValueError(f"scoring subjects the model was fitted on: {sorted(overlap)}")
    z = z_score(model, reps["dtw_dist"])
    score = rehabhub_score(z)
    missing = np.isnan(score)
    return pd.DataFrame({
        "schema_version": SCHEMA_VERSION,
        "rep_id": reps["rep_id"].to_numpy(),
        "scorer_id": SCORER_ID,
        "ref_type": ref_type,
        "mapping": MAPPING,
        "fold_id": fold_id,
        "score_0_100": score,
        "percentile": np.nan,          # this baseline has no percentile
        "subscores": np.nan,           # and no sub-scores
        "abstain": missing,
        "abstain_reason": np.where(missing, "no_dtw_distance", None),
        "feedback_code": None,
    }, columns=SCORE_COLUMNS)


def main():
    from fixtures.make_mock_reps import BLOCKS, make_mock_reps
    from neuroflow.scoring.folds import iter_folds, make_folds

    reps = make_mock_reps()
    folds = make_folds(reps["subject_id"])

    # Leave-one-subject-out: fit on the other development subjects, score the held-out one
    parts = []
    for fold_id, train, test in iter_folds(reps, folds):
        model = fit_baseline(train)
        parts.append(score_baseline(model, test, fold_id=fold_id))
    scores = pd.concat(parts, ignore_index=True)
    assert list(scores.columns) == SCORE_COLUMNS

    table = scores.merge(reps[["rep_id", "condition", "level"]], on="rep_id")
    order = list(dict.fromkeys((c, l) for _, c, l in BLOCKS))
    out = table.groupby(["condition", "level"])["score_0_100"].agg(
        n="size",
        median="median",
        at_100=lambda s: (s == 100).mean(),
    ).reindex(order)
    out["at_100"] = (100 * out["at_100"]).round(0).astype(int).astype(str) + " %"
    out["median"] = out["median"].round(1)
    print("RehabHub baseline under LOSO (mock data), per condition:\n")
    print(out.to_string())

    normal_100 = (table.loc[table["condition"] == NORMAL, "score_0_100"] == 100).mean()
    slow_100 = (table.loc[table["condition"] == "slow", "score_0_100"] == 100).mean()
    print(f"\nsaturation: {normal_100:.0%} of normal reps and {slow_100:.0%} of slow reps "
          f"score exactly 100 -> the score cannot tell slow reps from normal ones.")


if __name__ == "__main__":
    main()
