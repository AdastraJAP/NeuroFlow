"""
C3: scorer v1 (owner: Wilkend).
Standardise -> Ledoit-Wolf covariance -> Mahalanobis distance -> pooled ECDF -> 0-100.
Everything is fitted on NORMAL reps of TRAIN subjects only.

Run (from the repo root):
    python -m neuroflow.scoring.scorer
"""

import json

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from neuroflow.scoring.contract import IN_SAMPLE_REF_TYPE, SCHEMA_VERSION, SCORE_COLUMNS
from neuroflow.scoring.folds import check_not_fitted_on

SCORER_ID = "lw_mahalanobis_v1"
MAPPING = "lw_mahalanobis_ecdf"

FEATURES = [
    "dtw_dist", "duration_ratio", "rom_rel_personal",
    "time_to_peak_ratio", "sparc", "max_trunk_lean_deg", "min_elbow_deg",
]


def feature_matrix(reps: pd.DataFrame) -> pd.DataFrame:
    """Return one row per rep with the 7 feature columns above."""
    calib = reps[(reps["block_id"] == "normal_calib") & (reps["session_id"] == "s1")]
    med = calib.groupby("subject_id")["duration_s"].median().dropna()

    missing = set(reps["subject_id"].unique()) - set(med.index)
    if missing:
        raise ValueError(f"no session-1 normal_calib reps for subjects: {sorted(missing)}")

    out = reps.copy()
    out["duration_ratio"] = out["duration_s"] / out["subject_id"].map(med)
    return out[FEATURES]


def fit_scorer(train_reps: pd.DataFrame) -> dict:
    """Fit on NORMAL train reps only. Return everything score_reps needs."""
    normal = train_reps[train_reps["condition"] == "normal"]
    X = feature_matrix(normal).dropna()          # LedoitWolf cannot take NaN

    mu = X.mean()
    sd = X.std(ddof=1)
    Z = ((X - mu) / sd).to_numpy()               # standardise first

    prec = LedoitWolf().fit(Z).precision_        # this is the inverse of Sigma
    d = np.sqrt(np.einsum("ij,jk,ik->i", Z, prec, Z))

    from neuroflow.scoring.subscores import fit_subscores   # here: avoids a circular import

    return {
        "features": FEATURES,
        "mu": mu,
        "sd": sd,
        "prec": prec,
        "d_ref": np.sort(d),                     # sorted for np.searchsorted
        "n_ref": len(d),
        "train_subjects": sorted(train_reps["subject_id"].unique()),
        "sub": fit_subscores(train_reps),        # C4: per-feature ECDFs
    }


def mahalanobis(model: dict, reps: pd.DataFrame) -> np.ndarray:
    """Distance of every rep to the normal centre. NaN if a feature is missing."""
    X = feature_matrix(reps)
    Z = ((X - model["mu"]) / model["sd"]).to_numpy()   # the model's mu/sd: no refit
    d2 = np.einsum("ij,jk,ik->i", Z, model["prec"], Z)  # NaN rows stay NaN
    return np.sqrt(d2)


def score_reps(model: dict, reps: pd.DataFrame, fold_id=-1,
               ref_type="pooled", allow_in_sample=False) -> pd.DataFrame:
    """
    Score reps with a fitted model. Returns a scores table (SCORE_COLUMNS).
    allow_in_sample=True permits scoring training subjects and marks every row
    ref_type="in_sample", so in-sample rows can never pass for LOSO/locked ones.
    """
    check_not_fitted_on(model["train_subjects"], reps, allow_in_sample)
    if allow_in_sample:
        ref_type = IN_SAMPLE_REF_TYPE

    from neuroflow.scoring.subscores import deviation_table, feedback_codes, subscore_table

    d = mahalanobis(model, reps)
    missing = np.isnan(d)

    # ECDF: share of normal training reps that are at least as close as this rep
    p = np.searchsorted(model["d_ref"], np.nan_to_num(d), side="right") / model["n_ref"]
    p = np.where(missing, np.nan, p)
    score = 100.0 * (1.0 - p)

    # C4: one 0-100 sub-score per feature, plus a feedback code
    dev = deviation_table(model["sub"], reps)
    subs = subscore_table(model["sub"], reps, dev=dev)
    feedback = feedback_codes(score, subs, dev).to_numpy(copy=True)   # pandas 3: to_numpy() can be read-only
    feedback[missing] = None

    # Keep the distance itself: the score floors at 0 for every rep beyond the
    # most unusual normal training rep, d still grades how far out it is.
    # Abstained reps get no subscores at all (None), not {"d": null}.
    def as_json(i):
        if missing[i]:
            return None
        row = {"d": round(float(d[i]), 4)}
        for name in subs.columns:
            v = subs[name].iat[i]
            row[name] = None if np.isnan(v) else round(float(v), 2)
        return json.dumps(row)

    subscores = [as_json(i) for i in range(len(d))]

    return pd.DataFrame({
        "schema_version": SCHEMA_VERSION,
        "rep_id": reps["rep_id"].to_numpy(),
        "scorer_id": SCORER_ID,
        "ref_type": ref_type,
        "mapping": MAPPING,
        "fold_id": fold_id,
        "score_0_100": score,
        "percentile": p,
        "subscores": subscores,        # {"d": ..., "rom": ..., ..., "hike": ...}
        "abstain": missing,
        "abstain_reason": np.where(missing, "missing_feature", None),
        "feedback_code": feedback,
    }, columns=SCORE_COLUMNS)


def main():
    from fixtures.make_mock_reps import make_mock_reps
    from neuroflow.scoring.folds import iter_folds, make_folds

    reps = make_mock_reps()
    folds = make_folds(reps["subject_id"])

    parts = []
    for fold_id, train, test in iter_folds(reps, folds):
        model = fit_scorer(train)
        # Pass ALL of the test subject's reps: feature_matrix needs their
        # session-1 calibration block to compute duration_ratio.
        parts.append(score_reps(model, test, fold_id=fold_id))
    scores = pd.concat(parts, ignore_index=True)

    # Report session 2 only (calibrate in s1, test in s2)
    out = scores.merge(reps[["rep_id", "session_id", "condition", "level"]], on="rep_id")
    out = out[out["session_id"] == "s2"]
    table = (out.groupby(["condition", "level"])["score_0_100"]
                .median().round(1).rename("median score"))
    print("LOSO, session 2 reps, median score per condition:")
    print(table.to_string())

    normal = out.loc[out["condition"] == "normal", "score_0_100"].dropna()
    print(f"\nnormal reps: share scoring exactly 100 = {(normal == 100).mean():.1%} "
          f"(RehabHub baseline: ~83%)")
    print(f"abstained (missing feature): {int(scores['abstain'].sum())} reps")


if __name__ == "__main__":
    main()
