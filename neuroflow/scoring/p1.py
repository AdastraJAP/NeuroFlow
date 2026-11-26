"""
C8: P1 reproduction (owner: Wilkend).

P1 asks: does RehabHub-style scoring hold up without subject leakage? RehabHub
splits each participant's normal reps at random between the reference
(template / baseline) and the test-normal set, so every participant is on both
sides. We compare that split with leave-one-subject-out (LOSO), using the SAME
scorer and the SAME evaluation reps, so the only difference is the leak.

Within-participant split (RehabHub)
    For each development subject, its session-2 normal_test reps are shuffled
    (fixed seed) and halved: one half is the test-normal set, the other half
    and all its other normal reps go into the reference. One model is fitted on
    all subjects' reference reps and scores every development rep in-sample
    (allow_in_sample=True, ref_type="in_sample").

LOSO
    run_loso (C5): the held-out subject is never in the reference.

Evaluation (both arms, identical)
    test-normal half vs each condition's session-2 reps, abstained reps of either
    arm dropped. Gap = AUROC(within) - AUROC(LOSO), paired over the same subject
    resamples (C6). A positive gap is the optimism that leakage buys.

What the mock can and cannot show: mock reps carry a precomputed dtw_dist, so
the template cannot be rebuilt from the test subject's own reps. The leak here
only reaches the scorer's pooled statistics (mu/sigma for RehabHub; mean,
covariance and ECDF for C3). RehabHub's main leak, a template built partly from
the participant's own reps, needs the per-fold DTW template (B + C task).

Run (from the repo root):
    python -m neuroflow.scoring.p1
"""

import numpy as np
import pandas as pd

from neuroflow.scoring.roles import normal_test_mask, session_mask, test_mask

SEED = 0
EVAL_SESSION = "s2"       # the mock; pass session=None on real data (all sessions)


def within_split(reps, folds, seed=SEED, session=EVAL_SESSION):
    """
    rep_ids of the test-normal half: for each development subject, half of its
    session-2 normal_test reps (shuffled with a fixed seed). Everything else that
    is normal goes to the reference.
    """
    rng = np.random.default_rng(seed)
    dev = set(folds.loc[folds["split"] == "train", "subject_id"])
    pool = reps[reps["subject_id"].isin(dev) & session_mask(reps, session)
                & normal_test_mask(reps)]
    test_normal = []
    for _, g in pool.groupby("subject_id", sort=True):
        ids = rng.permutation(g["rep_id"].to_numpy())
        test_normal.extend(ids[: len(ids) // 2])
    return set(test_normal)


def run_within(reps, folds, fit, score, test_normal, curves=None, dtw_method="dba"):
    """
    Fit once on every development subject's reference reps, score all in-sample.
    With curves, the DTW template is built from those reference reps too, so it
    contains each person's own normal reps: RehabHub's main leak.
    """
    dev_subjects = set(folds.loc[folds["split"] == "train", "subject_id"])
    dev = reps[reps["subject_id"].isin(dev_subjects)]
    reference = dev[~dev["rep_id"].isin(test_normal)]
    if curves is not None:
        from neuroflow.segmentation.dtw_feature import add_dtw_dist
        dev = add_dtw_dist(dev, curves, reference, method=dtw_method)
        reference = dev[~dev["rep_id"].isin(test_normal)]
    model = fit(reference)
    return score(model, dev, fold_id=-1, allow_in_sample=True)


def eval_reps(reps, test_normal, session=EVAL_SESSION):
    """The reps both arms are judged on: the test-normal half + every test condition rep."""
    imp = reps[session_mask(reps, session) & (reps["condition"] != "normal") & test_mask(reps)]
    return set(test_normal) | set(imp["rep_id"])


def p1_table(reps, folds, fit, score, n_boot=2000, seed=SEED, session=EVAL_SESSION,
             curves=None, dtw_method="dba"):
    """One row per (condition, level): AUROC within, AUROC LOSO, paired gap, CIs."""
    from neuroflow.scoring.harness import POSITIVE_CONTROLS, run_loso
    from neuroflow.scoring.stats import paired_auroc_diff

    test_normal = within_split(reps, folds, seed, session)
    keep = eval_reps(reps, test_normal, session)
    within = run_within(reps, folds, fit, score, test_normal, curves, dtw_method)
    loso = run_loso(reps, folds, fit, score, curves=curves, dtw_method=dtw_method)

    pairs = (reps.loc[reps["rep_id"].isin(keep) & (reps["condition"] != "normal"),
                      ["condition", "level"]].drop_duplicates()
                 .sort_values(["condition", "level"]))
    rows = []
    for cond, lvl in pairs.itertuples(index=False):
        r = paired_auroc_diff(within, loso, reps, cond, lvl, n_boot=n_boot, seed=seed,
                              session=session, keep=keep, allow_in_sample=True)
        rows.append({"condition": cond, "level": lvl,
                     "within": r["a"], "within_lo": r["a_lo"], "within_hi": r["a_hi"],
                     "loso": r["b"], "loso_lo": r["b_lo"], "loso_hi": r["b_hi"],
                     "gap": r["diff"], "gap_lo": r["lo"], "gap_hi": r["hi"],
                     "gap_excludes_zero": r["excludes_zero"],
                     "n_subjects": r["n_subjects"],
                     "positive_control": cond in POSITIVE_CONTROLS})
    return pd.DataFrame(rows)


def main():
    from fixtures.make_mock_reps import make_mock_reps
    from neuroflow.scoring.baseline import fit_baseline, score_baseline
    from neuroflow.scoring.folds import make_folds
    from neuroflow.scoring.scorer import fit_scorer, score_reps

    print(f"seed = {SEED}; same scorer, same evaluation reps, within-participant vs LOSO\n")
    reps = make_mock_reps()
    folds = make_folds(reps["subject_id"])
    for name, fit, score in (("RehabHub (C2)", fit_baseline, score_baseline),
                             ("Scorer v1 (C3)", fit_scorer, score_reps)):
        t = p1_table(reps, folds, fit, score)
        out = pd.DataFrame({
            "condition": t["condition"] + " " + t["level"],
            "within": t.apply(lambda r: f"{r['within']:.3f} [{r['within_lo']:.3f}, {r['within_hi']:.3f}]", axis=1),
            "LOSO": t.apply(lambda r: f"{r['loso']:.3f} [{r['loso_lo']:.3f}, {r['loso_hi']:.3f}]", axis=1),
            "gap (within - LOSO)": t.apply(lambda r: f"{r['gap']:+.3f} [{r['gap_lo']:+.3f}, {r['gap_hi']:+.3f}]", axis=1),
            "gap?": np.where(t["gap_excludes_zero"], "yes", "no"),
            "control": t["positive_control"],
        })
        print(f"{name}, n = {t['n_subjects'].iloc[0]} subjects:")
        print(out.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
