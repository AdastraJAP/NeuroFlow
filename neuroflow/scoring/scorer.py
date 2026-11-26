"""
C3: scorer v1 (owner: Wilkend).
Standardise -> Ledoit-Wolf covariance -> Mahalanobis distance -> pooled ECDF -> 0-100.
Everything is fitted on NORMAL reps of TRAIN subjects only.

Real-data rules (issue #14):
  * calibration reps come from neuroflow.scoring.roles (is_calib / block_role /
    the mock's normal_calib), from each subject's baseline session;
  * a feature that is unavailable on the normal training reps (all NaN or
    constant: elbow and trunk on hip-free G1 clips) is dropped before fitting,
    and model["features"] lists the features actually used;
  * one model never mixes two angle definitions (angle_col) or two arms (side);
  * every abstained rep says why (contract.ABSTAIN_REASONS).

Run (from the repo root):
    python -m neuroflow.scoring.scorer
"""

import json

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from neuroflow.scoring.contract import IN_SAMPLE_REF_TYPE, SCHEMA_VERSION, SCORE_COLUMNS
from neuroflow.scoring.folds import check_not_fitted_on
from neuroflow.scoring.roles import calib_duration_median

SCORER_ID = "lw_mahalanobis_v1"
MAPPING = "lw_mahalanobis_ecdf"

FEATURES = [
    "dtw_dist", "duration_ratio", "rom_rel_personal",
    "time_to_peak_ratio", "sparc", "max_trunk_lean_deg", "min_elbow_deg",
]


def feature_matrix(reps: pd.DataFrame, missing_calib="raise") -> pd.DataFrame:
    """
    One row per rep with the FEATURES columns. A feature column the reps table
    does not have comes back all NaN (e.g. dtw_dist before add_dtw_dist).

    missing_calib: "raise" (fitting: every training subject needs a calibration)
    or "nan" (scoring: that subject's duration_ratio is NaN and its reps abstain
    with reason "no_calibration").
    """
    med = calib_duration_median(reps)
    missing = set(reps["subject_id"].unique()) - set(med.index)
    if missing and missing_calib == "raise":
        raise ValueError("no calibration reps with a duration (block_role 'calib' / "
                         f"block 'normal_calib') for subjects: {sorted(missing)}")

    out = pd.DataFrame(index=reps.index)
    for f in FEATURES:
        out[f] = reps[f].astype(float) if f in reps.columns else np.nan
    out["duration_ratio"] = reps["duration_s"] / reps["subject_id"].map(med)
    return out[FEATURES]


def _single(reps, col, what):
    """The single value of col in a table (None if absent); raise if it mixes two."""
    if col not in reps.columns:
        return None
    vals = reps[col].dropna().unique()
    if len(vals) > 1:
        raise ValueError(f"reps mix {what} {sorted(vals)}; fit one model per {col}")
    return vals[0] if len(vals) else None


def _angle(reps):
    return _single(reps, "angle_col", "angle definitions")


def fit_scorer(train_reps: pd.DataFrame) -> dict:
    """Fit on NORMAL train reps only. Return everything score_reps needs."""
    angle = _angle(train_reps)
    side = _single(train_reps, "side", "arms (side)")     # left and right never pooled
    normal = train_reps[train_reps["condition"] == "normal"]
    X = feature_matrix(normal)
    used = [f for f in FEATURES
            if X[f].notna().any() and X[f].std(ddof=1) > 0]   # unavailable -> dropped
    if not used:
        raise ValueError("no usable feature on the normal training reps")
    X = X[used].dropna()                          # LedoitWolf cannot take NaN
    if len(X) < 2 * len(used):
        raise ValueError(f"only {len(X)} complete normal training reps for "
                         f"{len(used)} features")

    mu = X.mean()
    sd = X.std(ddof=1)
    Z = ((X - mu) / sd).to_numpy()               # standardise first

    prec = LedoitWolf().fit(Z).precision_        # this is the inverse of Sigma
    d = np.sqrt(np.einsum("ij,jk,ik->i", Z, prec, Z))

    from neuroflow.scoring.subscores import fit_subscores   # here: avoids a circular import

    return {
        "features": used,
        "dropped_features": [f for f in FEATURES if f not in used],
        "angle_col": angle,
        "side": side,
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
    X = feature_matrix(reps, missing_calib="nan")[model["features"]]
    Z = ((X - model["mu"]) / model["sd"]).to_numpy()   # the model's mu/sd: no refit
    d2 = np.einsum("ij,jk,ik->i", Z, model["prec"], Z)  # NaN rows stay NaN
    return np.sqrt(d2)


def abstain_reasons(model: dict, reps: pd.DataFrame) -> np.ndarray:
    """
    Why each rep cannot be scored (None = it can). First matching reason wins:
    mixed_angle_col, mixed_side, no_calibration, censored_start, censored_end,
    missing_feature:<first missing used feature>.
    """
    used = model["features"]
    X = feature_matrix(reps, missing_calib="nan")[used]
    out = np.full(len(reps), None, dtype=object)
    free = lambda: np.array([r is None for r in out])          # noqa: E731
    if model.get("angle_col") is not None and "angle_col" in reps.columns:
        bad = (reps["angle_col"].notna() & reps["angle_col"].ne(model["angle_col"])).to_numpy()
        out[bad] = "mixed_angle_col"
    if model.get("side") is not None and "side" in reps.columns:
        bad = (reps["side"].notna() & reps["side"].ne(model["side"])).to_numpy()
        out[free() & bad] = "mixed_side"
    if "duration_ratio" in used:
        no_cal = ~reps["subject_id"].isin(calib_duration_median(reps).index).to_numpy()
        out[free() & no_cal] = "no_calibration"
    timing = {"duration_ratio", "time_to_peak_ratio", "sparc", "dtw_dist"} & set(used)
    for col, why in (("t_start_ms", "censored_start"), ("t_end_ms", "censored_end")):
        if timing and col in reps.columns:
            out[free() & reps[col].isna().to_numpy()] = why
    isna = X.isna().to_numpy()
    for i in np.flatnonzero(free() & isna.any(axis=1)):
        out[i] = f"missing_feature:{used[int(np.argmax(isna[i]))]}"
    return out


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
    reasons = abstain_reasons(model, reps)
    missing = np.isnan(d) | np.array([r is not None for r in reasons])
    d = np.where(missing, np.nan, d)

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
        "abstain_reason": np.where(missing, reasons, None),
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
    print(f"abstained: {int(scores['abstain'].sum())} reps, by reason:")
    print(scores["abstain_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
