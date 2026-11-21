"""
Per-fold DTW feature `dtw_dist` (owner: Member B, used by Member C's scorer).

Why per fold: the template is an average of NORMAL reps. If a held-out subject's
reps were in it, their own reps would look closer to "normal" than they are
(leakage). So `rep_features()` (B6) does not compute dtw_dist. The scorer builds
the template inside each fold, from the training subjects only.

Two steps:
  B's side, once per block (needs frames.parquet):
      curves = rep_curves(frames, reps, rest_ref, rom_ref)
      -> one row per rep_id, CURVE_COLUMNS = c00..c59: the rep cut out of the
         smoothed angle, normalised to the user's range ((theta - rest) / rom, so
         rest = 0 and the typical peak = 1) and stretched to TEMPLATE_LEN samples.
         Saved next to reps.parquet as rep_curves.parquet.
  C's side, inside each fold (no frames needed):
      reps = add_dtw_dist(reps, curves, train_reps)
      -> template from the NORMAL reps of train_reps only, then
         dtw_dist = banded DTW distance (B4) of every rep's curve to it.
         Same definition as the contract: sqrt of summed squared differences on
         the range-normalised angle, no unit. NaN when a rep has no curve.

Run (from the repo root):
    python -m neuroflow.segmentation.dtw_feature        (small per-fold demo)
"""

import numpy as np
import pandas as pd

from neuroflow.segmentation.dtw import dtw_distance
from neuroflow.segmentation.templates import TEMPLATE_LEN, dba_template, mean_template, rep_matrix

CURVE_COLUMNS = [f"c{i:02d}" for i in range(TEMPLATE_LEN)]
BAND_FRAC = 0.15            # same band as B5's templates


# ---- B's side: curves from frames ----

def rep_curves(frames, reps, rest_ref, rom_ref, angle_col=None):
    """One normalised, stretched curve per rep (index rep_id, columns CURVE_COLUMNS).
    Reps too short or too gappy to resample are left out (they get dtw_dist NaN)."""
    R, kept = rep_matrix(frames, reps, rest_ref, rom_ref, n=TEMPLATE_LEN, angle_col=angle_col)
    if len(kept) == 0:
        R = np.empty((0, TEMPLATE_LEN))
    return pd.DataFrame(R, index=pd.Index(kept, name="rep_id"), columns=CURVE_COLUMNS)


# ---- C's side: template and distance inside a fold ----

def fit_template(curves, train_reps, method="dba", band_frac=BAND_FRAC):
    """
    Template from the NORMAL reps of train_reps that have a curve.
    Pass the fold's training subjects only. method: "dba" (B5 default) or "mean".
    """
    ids = train_reps.loc[train_reps["condition"] == "normal", "rep_id"]
    R = curves.reindex(ids).dropna().to_numpy()
    if len(R) == 0:
        raise ValueError("no normal training reps with a curve to build the template from")
    if method == "mean":
        return mean_template(R)
    if method == "dba":
        return dba_template(R, band_frac=band_frac)[0]
    raise ValueError(f"method must be 'dba' or 'mean', got {method!r}")


def dtw_dist(curves, template, band_frac=BAND_FRAC):
    """Banded DTW distance of every curve to the template (Series by rep_id)."""
    d = [dtw_distance(x, template, band_frac) for x in curves.to_numpy()]
    return pd.Series(d, index=curves.index, name="dtw_dist", dtype=float)


def add_dtw_dist(reps, curves, train_reps, method="dba", band_frac=BAND_FRAC):
    """
    Return a copy of reps with a dtw_dist column, using a template fitted on the
    normal reps of train_reps (the fold's training subjects). reps may be the
    held-out subject's reps, or all reps; reps without a curve get NaN.
    Only curves of train_reps' normal reps enter the template, so the held-out
    subject never shapes its own reference.
    """
    template = fit_template(curves, train_reps, method, band_frac)
    have = curves.reindex(reps["rep_id"]).dropna()
    d = dtw_dist(have, template, band_frac)
    out = reps.copy()
    out["dtw_dist"] = out["rep_id"].map(d).astype(float)
    return out


def main():
    from neuroflow.segmentation.templates import timing_varied_reps

    # 6 subjects x 10 normal reps + 1 subject with capped (smaller) reps
    rows, curves = [], []
    for s in range(6):
        R = timing_varied_reps(k=10, seed=s)
        for k, x in enumerate(R):
            rows.append({"rep_id": f"sub{s}_{k}", "subject_id": f"sub{s}", "condition": "normal"})
            curves.append(x)
    for k, x in enumerate(timing_varied_reps(k=10, seed=99)):
        rows.append({"rep_id": f"cap_{k}", "subject_id": "sub6", "condition": "rom_cap"})
        curves.append(0.5 * x)
    reps = pd.DataFrame(rows)
    curves = pd.DataFrame(curves, index=pd.Index(reps["rep_id"], name="rep_id"), columns=CURVE_COLUMNS)

    for held in ("sub0", "sub6"):
        train = reps[reps["subject_id"] != held]
        out = add_dtw_dist(reps[reps["subject_id"] == held], curves, train)
        print(f"held out {held}: median dtw_dist {out['dtw_dist'].median():.3f}")


if __name__ == "__main__":
    main()
