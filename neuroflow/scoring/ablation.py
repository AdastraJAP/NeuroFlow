"""
C7 v0: P2 ablation grid (owner: Wilkend).

P2 asks which design choices fix DTW's blindness to slow, full-range reps. The
full grid is reference x alignment x tempo term x mapping (work.md, section 5).

What can run now, on mock reps: TEMPO TERM x MAPPING. The reference (pooled mean
/ DBA / own-session) and alignment (unconstrained / band 10 % / 20 %) factors
need DTW recomputed from angle curves with a template built per fold from the
training subjects (the shared B + C "per-fold DTW template" task, not started).
mock_reps.parquet only carries a precomputed dtw_dist, so those two factors are
held fixed here.

Factors in v0
    tempo term   none          shape only: dtw_dist
                 duration      + duration_ratio
                 duration+ttp  + duration_ratio + time_to_peak_ratio
    mapping      z100          RehabHub mapping on the cell's distance d:
                               Z = (d - mean_normal) / sd_normal, 100 / max(Z, 1)
                 lw_ecdf       C3 mapping: 100 x (1 - ECDF of normal-train d)
                 gmm           2-component Gaussian mixture on the standardised
                               features; 100 x (1 - ECDF of normal-train -loglik)

d = Ledoit-Wolf Mahalanobis distance on features standardised with normal-train
mean/SD (for one feature, d = |z|, so z100 x none is RehabHub made two-sided).
Every fit uses normal reps of training subjects only.

Run (from the repo root):
    python -m neuroflow.scoring.ablation
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

from neuroflow.scoring.contract import IN_SAMPLE_REF_TYPE
from neuroflow.scoring.folds import check_not_fitted_on

SEED = 0
TEMPO_TERMS = {
    "none":         ["dtw_dist"],
    "duration":     ["dtw_dist", "duration_ratio"],
    "duration+ttp": ["dtw_dist", "duration_ratio", "time_to_peak_ratio"],
}
MAPPINGS = ("z100", "lw_ecdf", "gmm")
GMM_COMPONENTS = 2           # fixed in advance; tune on training subjects only, later

# Full grid on the conditions P2 is about; point AUROCs only elsewhere.
FULL_GRID = (("slow", "2x"), ("slow", "3x"), ("band", "light"), ("band", "medium"))


def _ecdf_score(ref_sorted, x):
    x = np.asarray(x, dtype=float)
    p = np.searchsorted(ref_sorted, np.nan_to_num(x), side="right") / len(ref_sorted)
    return np.where(np.isnan(x), np.nan, 100.0 * (1.0 - p))


def make_cell(tempo, mapping):
    """(fit, score) for one grid cell, in the run_loso interface."""
    from neuroflow.scoring.scorer import feature_matrix

    feats = TEMPO_TERMS[tempo]
    if mapping not in MAPPINGS:
        raise ValueError(f"unknown mapping: {mapping}")

    def _z(model, reps):
        X = feature_matrix(reps)[feats]
        return ((X - model["mu"]) / model["sd"]).to_numpy()

    def _badness(model, Z):
        """Larger = more unusual. NaN rows stay NaN."""
        out = np.full(len(Z), np.nan)
        ok = ~np.isnan(Z).any(axis=1)
        if ok.any():
            if mapping == "gmm":
                out[ok] = -model["gmm"].score_samples(Z[ok])
            else:
                out[ok] = np.sqrt(np.einsum("ij,jk,ik->i", Z[ok], model["prec"], Z[ok]))
        return out

    def fit(train_reps):
        normal = train_reps[train_reps["condition"] == "normal"]
        X = feature_matrix(normal)[feats].dropna()
        model = {"mu": X.mean(), "sd": X.std(ddof=1),
                 "train_subjects": sorted(train_reps["subject_id"].unique())}
        Z = ((X - model["mu"]) / model["sd"]).to_numpy()
        if mapping == "gmm":
            model["gmm"] = GaussianMixture(GMM_COMPONENTS, covariance_type="full",
                                           random_state=SEED).fit(Z)
        else:
            model["prec"] = LedoitWolf().fit(Z).precision_
        b = _badness(model, Z)
        model["ref"] = np.sort(b)
        model["b_mean"], model["b_sd"] = float(b.mean()), float(b.std(ddof=1))
        return model

    def score(model, reps, fold_id=-1, allow_in_sample=False):
        check_not_fitted_on(model["train_subjects"], reps, allow_in_sample)
        b = _badness(model, _z(model, reps))
        if mapping == "z100":
            z = (b - model["b_mean"]) / model["b_sd"]
            s = 100.0 / np.maximum(z, 1.0)
        else:
            s = _ecdf_score(model["ref"], b)
        return pd.DataFrame({
            "rep_id": reps["rep_id"].to_numpy(),
            "fold_id": fold_id,
            "ref_type": IN_SAMPLE_REF_TYPE if allow_in_sample else "pooled",
            "score_0_100": s,
            "abstain": np.isnan(s),
        })

    return fit, score


def run_grid(reps, folds, n_boot=1000, seed=SEED):
    """
    One row per (tempo, mapping, condition, level). AUROC everywhere; a subject-
    bootstrap CI for the FULL_GRID conditions. Same seed in every cell, so cells
    are compared on the same resamples.
    """
    from neuroflow.scoring.harness import POSITIVE_CONTROLS, auroc_by_condition, run_loso
    from neuroflow.scoring.stats import bootstrap_auroc

    rows = []
    for tempo, mapping in itertools.product(TEMPO_TERMS, MAPPINGS):
        fit, score = make_cell(tempo, mapping)
        scores = run_loso(reps, folds, fit, score)
        table = auroc_by_condition(scores, reps)
        for r in table.itertuples(index=False):
            lo = hi = np.nan
            if (r.condition, r.level) in FULL_GRID:
                ci = bootstrap_auroc(scores, reps, r.condition, r.level, n_boot, seed)
                lo, hi = ci["lo"], ci["hi"]
            rows.append({"tempo": tempo, "mapping": mapping,
                         "condition": r.condition, "level": r.level,
                         "auroc": r.auroc, "lo": lo, "hi": hi,
                         "coverage": r.coverage,
                         "positive_control": r.condition in POSITIVE_CONTROLS})
    return pd.DataFrame(rows)


def main():
    from fixtures.make_mock_reps import make_mock_reps
    from neuroflow.scoring.folds import make_folds

    print(f"seed = {SEED}; reference and alignment held fixed (precomputed dtw_dist)\n")
    reps = make_mock_reps()
    folds = make_folds(reps["subject_id"])
    grid = run_grid(reps, folds)

    full = grid[grid.apply(lambda r: (r["condition"], r["level"]) in FULL_GRID, axis=1)].copy()
    full["cell"] = full.apply(lambda r: f"{r['auroc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]", axis=1)
    full["condition"] = full["condition"] + " " + full["level"]
    print("Full grid, AUROC [95% subject-bootstrap CI], LOSO, s2:")
    print(full.pivot_table(index=["tempo", "mapping"], columns="condition",
                           values="cell", aggfunc="first").to_string())

    rest = grid[~grid.index.isin(full.index)].copy()
    rest["condition"] = rest["condition"] + " " + rest["level"]
    print("\nReduced grid (point AUROC; rom_cap is a positive control):")
    print(rest.pivot_table(index=["tempo", "mapping"], columns="condition",
                           values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
