"""
C6: statistics (owner: Wilkend).

Every interval here resamples SUBJECTS, never reps: reps of one person are not
independent (same body, same camera, same day), so resampling reps would treat
~280 correlated reps as ~280 independent ones and give intervals that are far too
narrow. With ~14 development subjects the honest unit is the subject.

Ladder (work.md, section 5):
  v0  bootstrap_mean_ci      percentile bootstrap of a mean, subjects resampled
  v1  bootstrap_auroc        per-condition AUROC with a subject-bootstrap CI
  v2  paired_auroc_diff      scorer A - scorer B on the SAME resamples
      icc_2_1, retest_agreement   ICC(2,1), SEM and MDC95 between session 1 and 2

A subject drawn twice in one resample counts twice: groups are indexed by their
position in the draw, never re-grouped by subject_id (which would merge the two
copies back into one).

Run (from the repo root):
    python -m neuroflow.scoring.stats
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from neuroflow.scoring.contract import IN_SAMPLE_REF_TYPE
from neuroflow.scoring.roles import normal_test_mask, role_columns, session_mask, test_mask

N_BOOT = 2000
SEED = 0
ALPHA = 0.05


# ---------------------------------------------------------------- basics

def auroc(normal, impaired):
    """
    P(impaired rep scores lower than a normal rep), ties count 1/2.
    Mann-Whitney U from average ranks: same value as roc_auc_score(y, -score)
    with y = 1 for impaired, but fast enough to call thousands of times.
    """
    normal = np.asarray(normal, dtype=float)
    impaired = np.asarray(impaired, dtype=float)
    n0, n1 = len(normal), len(impaired)
    if n0 == 0 or n1 == 0:
        return np.nan
    ranks = rankdata(np.concatenate([normal, impaired]))
    u = ranks[:n0].sum() - n0 * (n0 + 1) / 2.0
    return float(u / (n0 * n1))


def percentile_ci(values, alpha=ALPHA):
    """Percentile interval of bootstrap replicates (NaN replicates dropped)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))


def subject_draws(n_subjects, n_boot=N_BOOT, seed=SEED):
    """(n_boot, n_subjects) positions drawn with replacement. Fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_subjects, size=(n_boot, n_subjects))


# ---------------------------------------------------------------- v0

def bootstrap_mean_ci(values, subject_ids, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """
    Mean over all reps, with a CI from resampling subjects (with all their reps).
    Returns dict(estimate, lo, hi, n_subjects).
    """
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "s": np.asarray(subject_ids)})
    df = df.dropna(subset=["v"])
    groups = [g["v"].to_numpy() for _, g in df.groupby("s", sort=True)]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    draws = subject_draws(len(groups), n_boot, seed)
    reps_means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = percentile_ci(reps_means, alpha)
    return {"estimate": float(df["v"].mean()), "lo": lo, "hi": hi,
            "n_subjects": len(groups)}


# ---------------------------------------------------------------- v1

def _check_no_in_sample(scores):
    if "ref_type" in scores and scores["ref_type"].eq(IN_SAMPLE_REF_TYPE).any():
        raise ValueError("in-sample scores cannot go into a CI table; "
                         "use them only in the explicit P1 comparison")


def _labelled(scores, reps, session):
    cols = ["rep_id", "subject_id", *[c for c in role_columns(reps) if c != "subject_id"]]
    df = scores.merge(reps[cols], on="rep_id", how="left", validate="one_to_one")
    return df[session_mask(df, session)]


def subject_groups(scores, reps, condition, level, session="s2", keep=None,
                   allow_in_sample=False):
    """
    Per subject: (normal_test scores, scores of this condition/level), session only,
    abstained reps dropped. keep: optional set of rep_ids to restrict to (used to
    compare two scorers on exactly the same reps).
    allow_in_sample=True only for the explicit P1 comparison (C8).
    Returns (subject list, list of normal arrays, list of impaired arrays).
    """
    if not allow_in_sample:
        _check_no_in_sample(scores)
    df = _labelled(scores, reps, session)
    df = df[~df["abstain"].astype(bool)]
    if keep is not None:
        df = df[df["rep_id"].isin(keep)]
    is_normal = normal_test_mask(df)
    is_imp = (df["condition"] == condition) & (df["level"] == level) & test_mask(df)
    subjects = sorted(set(df.loc[is_normal | is_imp, "subject_id"]))
    normal, impaired = [], []
    for s in subjects:
        m = df["subject_id"] == s
        normal.append(df.loc[m & is_normal, "score_0_100"].to_numpy())
        impaired.append(df.loc[m & is_imp, "score_0_100"].to_numpy())
    return subjects, normal, impaired


def _boot_auroc(normal, impaired, draws):
    out = np.empty(len(draws))
    for b, idx in enumerate(draws):
        out[b] = auroc(np.concatenate([normal[i] for i in idx]),
                       np.concatenate([impaired[i] for i in idx]))
    return out


def bootstrap_auroc(scores, reps, condition, level, n_boot=N_BOOT, seed=SEED,
                    alpha=ALPHA, session="s2"):
    """AUROC of normal_test vs one condition/level, with a subject-bootstrap CI."""
    subjects, normal, impaired = subject_groups(scores, reps, condition, level, session)
    est = auroc(np.concatenate(normal), np.concatenate(impaired)) if subjects else np.nan
    lo, hi = percentile_ci(_boot_auroc(normal, impaired,
                                       subject_draws(len(subjects), n_boot, seed)), alpha)
    return {"auroc": est, "lo": lo, "hi": hi, "n_subjects": len(subjects)}


# ---------------------------------------------------------------- v2

def paired_auroc_diff(scores_a, scores_b, reps, condition, level, n_boot=N_BOOT,
                      seed=SEED, alpha=ALPHA, session="s2", keep=None,
                      allow_in_sample=False):
    """
    AUROC(A) - AUROC(B) on the same reps and the same subject resamples.

    Same reps: only reps that neither scorer abstained on, so a scorer cannot
    win by abstaining on hard reps. Same resamples: both AUROCs move together
    with the subjects drawn (they are positively correlated), so the variance of
    the difference is much smaller than the two separate CIs suggest. Comparing
    two separate CIs ("they overlap") is too conservative.
    """
    ok_a = set(scores_a.loc[~scores_a["abstain"].astype(bool), "rep_id"])
    ok_b = set(scores_b.loc[~scores_b["abstain"].astype(bool), "rep_id"])
    both = ok_a & ok_b
    keep = both if keep is None else both & set(keep)
    sa, na, ia = subject_groups(scores_a, reps, condition, level, session, keep,
                                allow_in_sample)
    sb, nb, ib = subject_groups(scores_b, reps, condition, level, session, keep,
                                allow_in_sample)
    assert sa == sb, "both scorers must cover the same subjects"
    draws = subject_draws(len(sa), n_boot, seed)
    boot_a, boot_b = _boot_auroc(na, ia, draws), _boot_auroc(nb, ib, draws)
    diff = boot_a - boot_b
    est = (auroc(np.concatenate(na), np.concatenate(ia))
           - auroc(np.concatenate(nb), np.concatenate(ib)))
    lo, hi = percentile_ci(diff, alpha)
    a_lo, a_hi = percentile_ci(boot_a, alpha)
    b_lo, b_hi = percentile_ci(boot_b, alpha)
    return {"diff": est, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "a": auroc(np.concatenate(na), np.concatenate(ia)), "a_lo": a_lo, "a_hi": a_hi,
            "b": auroc(np.concatenate(nb), np.concatenate(ib)), "b_lo": b_lo, "b_hi": b_hi,
            "n_subjects": len(sa), "n_reps": len(keep)}


def icc_2_1(matrix):
    """
    ICC(2,1), Shrout & Fleiss 1979: two-way random effects, absolute agreement,
    single measurement. matrix: n subjects x k sessions (or raters), no NaN.
    """
    x = np.asarray(matrix, dtype=float)
    if np.isnan(x).any():
        raise ValueError("ICC needs a complete subjects x sessions matrix")
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((x - grand) ** 2).sum() - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def retest_agreement(scores, reps, sessions=("s1", "s2")):
    """
    Agreement of each subject's mean score between two sessions.
    ICC(2,1); SEM = SD x sqrt(1 - ICC); MDC95 = 1.96 x sqrt(2) x SEM
    (the smallest change in a subject's mean score that is more than noise).

    MDC in points can only be compared between scores with similar spreads. A
    score that saturates (RehabHub: most normal reps at exactly 100) squeezes
    the between-subject SD, so its MDC in points shrinks without it being more
    reliable. So we also report share_at_100 and mdc95_rel = MDC95 / SD, which
    equals 1.96 x sqrt(2 x (1 - ICC)): compare scores on ICC or mdc95_rel.
    """
    _check_no_in_sample(scores)
    cols = ["rep_id", "subject_id", *[c for c in role_columns(reps) if c != "subject_id"]]
    df = scores.merge(reps[cols], on="rep_id", how="left", validate="one_to_one")
    df = df[normal_test_mask(df) & df["session_id"].isin(sessions)
            & ~df["abstain"].astype(bool)]
    m = df.pivot_table(index="subject_id", columns="session_id",
                       values="score_0_100", aggfunc="mean")[list(sessions)].dropna()
    icc = icc_2_1(m.to_numpy())
    sd = float(np.std(m.to_numpy(), ddof=1))
    sem = sd * np.sqrt(max(0.0, 1.0 - icc))
    mdc95 = 1.96 * np.sqrt(2.0) * sem
    return {"icc": icc, "sem": sem, "mdc95": mdc95, "sd": sd,
            "mdc95_rel": mdc95 / sd if sd > 0 else np.nan,
            "share_at_100": float((df["score_0_100"] >= 100).mean()),
            "n_subjects": len(m), "per_subject": m}


# ---------------------------------------------------------------- tables

def ci_table(scores, reps, n_boot=N_BOOT, seed=SEED, session="s2"):
    """bootstrap_auroc for every (condition, level) except normal."""
    df = _labelled(scores, reps, session)
    pairs = (df.loc[(df["condition"] != "normal") & test_mask(df), ["condition", "level"]]
               .drop_duplicates().sort_values(["condition", "level"]))
    rows = []
    for cond, lvl in pairs.itertuples(index=False):
        r = bootstrap_auroc(scores, reps, cond, lvl, n_boot, seed, session=session)
        rows.append({"condition": cond, "level": lvl, **r})
    return pd.DataFrame(rows)


def main():
    from fixtures.make_mock_reps import make_mock_reps
    from neuroflow.scoring.baseline import fit_baseline, score_baseline
    from neuroflow.scoring.folds import make_folds
    from neuroflow.scoring.harness import POSITIVE_CONTROLS, run_loso
    from neuroflow.scoring.scorer import fit_scorer, score_reps

    print(f"seed = {SEED}, n_boot = {N_BOOT}, subjects resampled (never reps)\n")
    reps = make_mock_reps()
    folds = make_folds(reps["subject_id"])
    base = run_loso(reps, folds, fit_baseline, score_baseline)
    c3 = run_loso(reps, folds, fit_scorer, score_reps)

    tb, t3 = ci_table(base, reps), ci_table(c3, reps)
    fmt = lambda r: f"{r['auroc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
    rows = []
    for (_, b), (_, c) in zip(tb.iterrows(), t3.iterrows()):
        d = paired_auroc_diff(c3, base, reps, b["condition"], b["level"])
        rows.append({
            "condition": b["condition"], "level": b["level"],
            "RehabHub AUROC [95% CI]": fmt(b), "C3 AUROC [95% CI]": fmt(c),
            "C3 - RehabHub [95% CI]": f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]",
            "difference?": "yes" if d["excludes_zero"] else "no",
            "control": b["condition"] in POSITIVE_CONTROLS,
        })
    print("LOSO, s2, normal_test vs each condition (n = "
          f"{tb['n_subjects'].iloc[0]} subjects):")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nTest-retest of each subject's mean normal_test score, s1 vs s2:")
    for name, sc in (("RehabHub", base), ("C3", c3)):
        r = retest_agreement(sc, reps)
        print(f"  {name:<9} ICC(2,1) = {r['icc']:.3f}   MDC95 = {r['mdc95']:.1f} points "
              f"(SD of subject means {r['sd']:.1f}; MDC95/SD = {r['mdc95_rel']:.2f})   "
              f"reps at exactly 100: {r['share_at_100']:.0%}   (n = {r['n_subjects']})")
    print("  MDC in points is not comparable across scores with different spreads "
          "(a ceiling shrinks it); compare ICC or MDC95/SD.")


if __name__ == "__main__":
    main()
