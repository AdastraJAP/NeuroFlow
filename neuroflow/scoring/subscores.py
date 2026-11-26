"""
C4: compensation sub-scores and feedback codes (owner: Wilkend).

C3 says "this rep is unusual"; C4 says why. Each sub-score is a 1-D pooled ECDF
on one feature, fitted on NORMAL reps of TRAIN subjects only (same rule as C3).

Direction of "worse" per feature:
    high  bigger is worse         (dtw_dist, trunk lean, hike)
    low   smaller is worse        (rom, sparc, elbow angle)
    two   far from normal either way (tempo: too slow or too fast)

rom is one-sided on purpose: lifting higher than calibration is progress, not an
error, and overshooting by leaning is already caught by the trunk sub-score.

Every feature is turned into a signed, robust deviation
    dev = (x - median) / scale     (scale = 1.4826 * MAD of normal train reps)
flipped so that a larger dev is always worse (|dev| for "two"). The sub-score is
100 * (1 - ECDF of the normal-train devs). dev is kept because it breaks ties
when several sub-scores floor at 0.

Run (from the repo root):
    python -m neuroflow.scoring.subscores
"""

import numpy as np
import pandas as pd

from neuroflow.scoring.contract import FEEDBACK_CODES
from neuroflow.scoring.scorer import feature_matrix

# sub-score -> (feature column, direction)
SUBSCORE_SPEC = {
    "rom":    ("rom_rel_personal",   "low"),
    "tempo":  ("duration_ratio",     "two"),
    "shape":  ("dtw_dist",           "high"),
    "smooth": ("sparc",              "low"),
    "trunk":  ("max_trunk_lean_deg", "high"),
    "elbow":  ("min_elbow_deg",      "low"),
    "hike":   ("max_hike_ratio",     "high"),    # exploratory: never a feedback code
    # exploratory: never a feedback code. Resting tilt = median of the 0.5 s just
    # before the rep's start (cut at the previous rep's end, >= 8 valid frames,
    # else NaN); feature = max |tilt - resting tilt| over the rep (B, issue #14).
    "tilt":   ("max_shoulder_tilt_deg", "high"),
}

# Feedback is given only when the rep is unusual overall AND one sub-score is low.
# Both thresholds are placeholders to be chosen on TRAIN subjects (work.md, sec. 8).
GLOBAL_THRESHOLD = 5.0
SUB_THRESHOLD = 5.0

MAD_TO_SD = 1.4826


def sub_features(reps: pd.DataFrame, missing_calib="raise") -> pd.DataFrame:
    """The feature column of every sub-score (reuses C3's feature_matrix).
    Columns the reps table lacks come back all NaN."""
    X = feature_matrix(reps, missing_calib)      # duration_ratio + calib check for free
    for col in ("max_hike_ratio", "max_shoulder_tilt_deg"):
        X[col] = reps[col].astype(float).to_numpy() if col in reps.columns else np.nan
    return X[[feat for feat, _ in SUBSCORE_SPEC.values()]]


def _robust_scale(x: np.ndarray) -> float:
    """1.4826 * MAD; falls back to SD, then to 1.0, if the spread is zero."""
    med = np.median(x)
    scale = MAD_TO_SD * np.median(np.abs(x - med))
    if not scale > 0:
        scale = np.std(x, ddof=1) if len(x) > 1 else 0.0
    if not scale > 0:
        scale = 1.0          # constant feature: any change is already extreme
    return float(scale)


def _deviation(x, med, scale, direction):
    """Signed robust deviation, oriented so that larger = worse."""
    z = (np.asarray(x, dtype=float) - med) / scale
    if direction == "high":
        return z
    if direction == "low":
        return -z
    if direction == "two":
        return np.abs(z)
    raise ValueError(f"unknown direction: {direction}")


def _ecdf_score(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """100 * (1 - ECDF). NaN in -> NaN out."""
    x = np.asarray(x, dtype=float)
    p = np.searchsorted(ref_sorted, np.nan_to_num(x), side="right") / len(ref_sorted)
    return np.where(np.isnan(x), np.nan, 100.0 * (1.0 - p))


def fit_subscores(train_reps: pd.DataFrame) -> dict:
    """Median, scale and sorted normal-train deviations per sub-score.
    A sub-score whose feature has no normal training value is left out."""
    normal = train_reps[train_reps["condition"] == "normal"]
    X = sub_features(normal)
    model = {}
    for name, (feat, direction) in SUBSCORE_SPEC.items():
        x = X[feat].dropna().to_numpy()
        if len(x) == 0:
            continue                       # unavailable on this data
        med, scale = float(np.median(x)), _robust_scale(x)
        model[name] = {
            "feature": feat,
            "direction": direction,
            "median": med,
            "scale": scale,
            "ref": np.sort(_deviation(x, med, scale, direction)),
        }
    return model


def deviation_table(model: dict, reps: pd.DataFrame) -> pd.DataFrame:
    """One robust-deviation column per sub-score (larger = worse)."""
    X = sub_features(reps, missing_calib="nan")
    return pd.DataFrame(
        {name: _deviation(X[m["feature"]], m["median"], m["scale"], m["direction"])
         for name, m in model.items()},
        index=reps.index)


def subscore_table(model: dict, reps: pd.DataFrame, dev=None) -> pd.DataFrame:
    """One 0-100 column per sub-score."""
    if dev is None:
        dev = deviation_table(model, reps)
    return pd.DataFrame(
        {name: _ecdf_score(model[name]["ref"], dev[name].to_numpy()) for name in model},
        index=dev.index)


def feedback_codes(score_0_100, subs: pd.DataFrame, dev: pd.DataFrame) -> pd.Series:
    """
    None, or the worst sub-score among FEEDBACK_CODES (hike excluded), given only
    when the global score AND that sub-score are both below their thresholds.
    Ties (e.g. several at 0) go to the largest deviation.
    """
    score_0_100 = np.asarray(score_0_100, dtype=float)
    codes = [c for c in FEEDBACK_CODES if c in subs.columns]
    S = subs[codes].to_numpy()
    D = dev[codes].to_numpy()

    out = []
    if not codes:
        return pd.Series([None] * len(subs), index=subs.index, dtype=object)
    for i in range(len(S)):
        if not score_0_100[i] < GLOBAL_THRESHOLD:     # also False for NaN
            out.append(None)
            continue
        ok = ~np.isnan(S[i])
        if not ok.any():
            out.append(None)
            continue
        # lowest sub-score first, then largest deviation
        order = sorted(np.flatnonzero(ok), key=lambda j: (S[i, j], -D[i, j]))
        j = order[0]
        out.append(codes[j] if S[i, j] < SUB_THRESHOLD else None)
    return pd.Series(out, index=subs.index, dtype=object)


def main():
    from fixtures.make_mock_reps import make_mock_reps
    from neuroflow.scoring.folds import iter_folds, make_folds
    from neuroflow.scoring.scorer import fit_scorer, score_reps

    reps = make_mock_reps()
    folds = make_folds(reps["subject_id"])
    parts = [score_reps(fit_scorer(train), test, fold_id=fold_id)
             for fold_id, train, test in iter_folds(reps, folds)]
    scores = pd.concat(parts, ignore_index=True)
    out = scores.merge(reps[["rep_id", "session_id", "condition", "level"]], on="rep_id")
    out = out[out["session_id"] == "s2"]

    print("LOSO, session 2: most common feedback code per condition")
    for (cond, lvl), g in out.groupby(["condition", "level"]):
        given = g["feedback_code"].dropna()
        top = given.value_counts().idxmax() if len(given) else "-"
        print(f"  {cond:<11} {lvl:<7} feedback on {len(given) / len(g):6.1%}   most common: {top}")

    normal = out[(out["condition"] == "normal") & ~out["abstain"]]
    print(f"\nfalse-feedback rate on held-out normal reps: "
          f"{normal['feedback_code'].notna().mean():.1%} "
          f"(thresholds: global < {GLOBAL_THRESHOLD}, sub-score < {SUB_THRESHOLD})")


if __name__ == "__main__":
    main()
