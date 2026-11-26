"""
C5: leave-one-subject-out harness (owner: Wilkend).

Answers the P1/P2 question: does a score separate normal reps from impaired reps
on a person it has never seen? Works with any scorer that follows

    fit(train_reps) -> model
    score(model, test_reps, fold_id) -> scores table (rep_id, score_0_100, abstain)

so the RehabHub baseline (C2) and the LW-Mahalanobis scorer (C3) plug in unchanged.

Evaluation rules (work.md, section 6):
  * normal = the normal_test block only. normal_calib reps define the personal
    median, so their duration ratio is ~1 by construction and would flatter.
  * session 2 only, for both classes: calibrate in s1, test in s2, and a session
    effect cannot pass for a condition effect.
  * one AUROC per (condition, level) against normal, never pooled.
  * rom_cap is a positive control: reported, flagged, never in a headline number.
  * abstained reps are excluded from the AUROC, so coverage is reported next to it.

Run (from the repo root):
    python -m neuroflow.scoring.harness
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from neuroflow.scoring.contract import IN_SAMPLE_REF_TYPE
from neuroflow.scoring.roles import (calib_mask, normal_test_mask, role_columns,
                                     session_mask, test_mask)

POSITIVE_CONTROLS = ("rom_cap",)
# Which reps are calibration / normal test / test: neuroflow.scoring.roles
# (block_role in schema v2, the mock's block names in v1).

# Leak test: decided BEFORE looking at any result.
SHUFFLE_TOLERANCE = 0.05          # mean shuffled AUROC must lie in 0.5 +/- 0.05

AUROC_COLUMNS = ["condition", "level", "n_normal", "n_impaired", "n_abstained",
                 "coverage", "auroc", "positive_control", "abstain_reasons"]


# ---------------------------------------------------------------- LOSO

def run_loso(reps, folds, fit, score, leak=False, curves=None, dtw_method="dba"):
    """
    Fit on the train subjects of every fold, score the held-out subject, and
    return all folds' scores concatenated (one row per development rep).

    leak=True is ONLY for the leak test's positive control: the model is fitted
    on train AND test subjects, then scored with allow_in_sample=True, because
    score_reps and score_baseline (rightly) refuse to score a training subject.
    Those rows come back with ref_type="in_sample".

    curves: B's rep_curves table (index rep_id, c00..c59). When given, dtw_dist
    is recomputed INSIDE each fold with add_dtw_dist: the template is built from
    the normal reps of that fold's training subjects only, so a held-out subject
    never shapes its own reference (issue #14). Without curves, the reps'
    precomputed dtw_dist is used (the mock).
    Known bias, same in every fold: the training reps are scored against a
    template they helped build, so their dtw_dist is slightly optimistic.
    """
    from neuroflow.scoring.folds import iter_folds

    parts = []
    for fold_id, train, test in iter_folds(reps, folds):
        if curves is not None:
            from neuroflow.segmentation.dtw_feature import add_dtw_dist
            source = pd.concat([train, test]) if leak else train
            train = add_dtw_dist(train, curves, source, method=dtw_method)
            test = add_dtw_dist(test, curves, source, method=dtw_method)
        if leak:
            model = fit(pd.concat([train, test]))
            parts.append(score(model, test, fold_id=fold_id,
                               allow_in_sample=True))      # deliberate leak
        else:
            model = fit(train)
            parts.append(score(model, test, fold_id=fold_id))
    return pd.concat(parts, ignore_index=True)


def _attach(scores, reps):
    cols = ["rep_id", "subject_id", *[c for c in role_columns(reps) if c != "subject_id"]]
    return scores.merge(reps[cols], on="rep_id", how="left", validate="one_to_one")


def _auroc(normal_scores, impaired_scores):
    """P(impaired rep scores lower than a normal rep); ties count 1/2."""
    if len(normal_scores) == 0 or len(impaired_scores) == 0:
        return np.nan
    y = np.r_[np.zeros(len(normal_scores)), np.ones(len(impaired_scores))]
    s = np.r_[normal_scores, impaired_scores]
    return roc_auc_score(y, -s)          # minus: a lower score means more impaired


def _reason_counts(g):
    if "abstain_reason" not in g.columns:
        return ""
    v = g.loc[g["abstain"].astype(bool), "abstain_reason"].fillna("unknown").value_counts()
    return ", ".join(f"{k}: {n}" for k, n in v.items())


def auroc_by_condition(scores, reps, session="s2"):
    """
    One row per (condition, level): normal_test vs that condition, one session.

    n_normal, n_impaired  reps that got a score (abstained reps excluded)
    n_abstained           abstained reps of BOTH classes: the normal_test reps
                          (same in every row) plus this condition's reps
    coverage              scored / all, over both classes
    abstain_reasons       counts per abstain_reason, both classes
    session=None uses every session (real data: test reps of all sessions).
    """
    if "ref_type" in scores and scores["ref_type"].eq(IN_SAMPLE_REF_TYPE).any():
        raise ValueError("in-sample scores in an AUROC table; use them only in the "
                         "explicit P1/leak comparisons")
    df = _attach(scores, reps)
    df = df[session_mask(df, session)]

    normal = df[normal_test_mask(df)]
    normal_ok = normal[~normal["abstain"]]

    rows = []
    impaired_all = df[(df["condition"] != "normal") & test_mask(df)]
    for (cond, lvl), g in impaired_all.groupby(["condition", "level"], sort=True):
        g_ok = g[~g["abstain"]]
        n_ab = int(g["abstain"].sum() + normal["abstain"].sum())
        n_tot = len(g) + len(normal)
        rows.append({
            "condition": cond,
            "level": lvl,
            "n_normal": len(normal_ok),
            "n_impaired": len(g_ok),
            "n_abstained": n_ab,
            "coverage": (n_tot - n_ab) / n_tot if n_tot else np.nan,
            "auroc": _auroc(normal_ok["score_0_100"].to_numpy(),
                            g_ok["score_0_100"].to_numpy()),
            "positive_control": cond in POSITIVE_CONTROLS,
            "abstain_reasons": _reason_counts(pd.concat([normal, g])),
        })
    return pd.DataFrame(rows, columns=AUROC_COLUMNS)


# ---------------------------------------------------------------- leak test

def shuffle_within_subject(reps, seed=0):
    """
    Permute (condition, level) together, inside each subject, over the
    non-calibration reps. The calibration block is left alone: it is protocol
    infrastructure (it sets the personal median), not a label.
    """
    rng = np.random.default_rng(seed)
    out = reps.copy()
    movable = ~calib_mask(out)
    for _, idx in out[movable].groupby("subject_id").groups.items():
        idx = np.asarray(idx)
        perm = rng.permutation(idx)
        out.loc[idx, ["condition", "level"]] = out.loc[perm, ["condition", "level"]].to_numpy()
    return out


def pooled_shuffled_auroc(scores, shuffled_reps, session="s2"):
    """
    One AUROC: every non-normal rep vs every normal rep, by the SHUFFLED labels.
    Pooled on purpose: under the null all conditions are exchangeable, so pooling
    only adds data, while 12 per-condition AUROCs would give 12 chances to cross
    the tolerance by luck. (The never-pool rule is for real effects, not this.)
    """
    df = _attach(scores, shuffled_reps)
    df = df[session_mask(df, session) & ~calib_mask(df) & ~df["abstain"].astype(bool)]
    is_normal = df["condition"] == "normal"
    return _auroc(df.loc[is_normal, "score_0_100"].to_numpy(),
                  df.loc[~is_normal, "score_0_100"].to_numpy())


def shuffled_label_check(reps, folds, fit, score, n_shuffles=5, seed=0, leak=False):
    """
    Shuffle labels BEFORE run_loso, so the shuffled labels flow through fitting.
    A held-out subject then carries no information about its own labels, so the
    AUROC against those labels must be ~0.5. A fit that peeks at the test
    subject (leak=True) can push it above 0.5. Returns one AUROC per shuffle.
    """
    out = []
    for k in range(n_shuffles):
        shuffled = shuffle_within_subject(reps, seed=seed + k)
        scores = run_loso(shuffled, folds, fit, score, leak=leak)
        out.append(pooled_shuffled_auroc(scores, shuffled))
    return np.array(out)


# ---------------------------------------------------------------- 1-NN control scorer
# NOT a real scorer: it exists only as the positive control of the leak test
# (main() and tests/test_harness.py). Its output has only the columns the harness
# needs (rep_id, fold_id, score_0_100, abstain), not the full SCORE_COLUMNS.

def fit_nearest_normal(train_reps):
    """
    Test-only scorer that MEMORISES: distance to the nearest normal training rep.
    A training rep is at distance 0 from itself, so if test reps leak into the
    reference they look perfect. That makes it the positive control for the leak
    test; a 7-mean Mahalanobis model is too smooth to memorise individual reps.
    """
    from neuroflow.scoring.scorer import feature_matrix

    X = feature_matrix(train_reps[train_reps["condition"] == "normal"]).dropna()
    mu, sd = X.mean(), X.std(ddof=1)
    return {"mu": mu, "sd": sd, "ref": ((X - mu) / sd).to_numpy()}


def score_nearest_normal(model, reps, fold_id=-1, allow_in_sample=False):
    # No training-subject check here on purpose: this scorer only exists to leak.
    from neuroflow.scoring.scorer import feature_matrix

    Z = ((feature_matrix(reps) - model["mu"]) / model["sd"]).to_numpy()
    missing = np.isnan(Z).any(axis=1)
    d = np.full(len(Z), np.nan)
    ok = ~missing
    if ok.any():
        diff = Z[ok][:, None, :] - model["ref"][None, :, :]
        d[ok] = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
    return pd.DataFrame({
        "rep_id": reps["rep_id"].to_numpy(),
        "fold_id": fold_id,
        "ref_type": IN_SAMPLE_REF_TYPE if allow_in_sample else "pooled",
        "score_0_100": 100.0 / (1.0 + d),   # monotone: closer = higher, in (0, 100]
        "abstain": missing,
    })


# ---------------------------------------------------------------- main

def main():
    from fixtures.make_mock_reps import make_mock_reps
    from neuroflow.scoring.baseline import fit_baseline, score_baseline
    from neuroflow.scoring.folds import make_folds
    from neuroflow.scoring.scorer import fit_scorer, score_reps

    reps = make_mock_reps()
    folds = make_folds(reps["subject_id"])

    base = auroc_by_condition(run_loso(reps, folds, fit_baseline, score_baseline), reps)
    c3 = auroc_by_condition(run_loso(reps, folds, fit_scorer, score_reps), reps)

    table = base[["condition", "level", "positive_control"]].copy()
    table["RehabHub (C2)"] = base["auroc"].round(3)
    table["C3"] = c3["auroc"].round(3)
    table["C3 coverage"] = c3["coverage"].round(3)
    print("LOSO, session 2, normal_test vs each condition: AUROC")
    print(table.to_string(index=False))
    print("(positive_control = True: rom_cap, never part of a headline number)\n")

    print(f"Leak test: labels shuffled within subject before fitting, "
          f"pass if mean is 0.5 +/- {SHUFFLE_TOLERANCE}")
    checks = [
        ("C3 scorer, LOSO", fit_scorer, score_reps, False),
        ("1-NN, LOSO", fit_nearest_normal, score_nearest_normal, False),
        ("C3 scorer, LEAKY (control)", fit_scorer, score_reps, True),
        ("1-NN, LEAKY (positive control)", fit_nearest_normal, score_nearest_normal, True),
    ]
    for name, fit, score, leak in checks:
        a = shuffled_label_check(reps, folds, fit, score, n_shuffles=5, seed=0, leak=leak)
        ok = abs(a.mean() - 0.5) <= SHUFFLE_TOLERANCE
        print(f"  {name:<32} mean {a.mean():.3f}  "
              f"[{', '.join(f'{x:.3f}' for x in a)}]  {'pass' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
