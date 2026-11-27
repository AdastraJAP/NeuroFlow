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
         rest = 0 and the typical peak = 1) and stretched to TEMPLATE_LEN samples,
         plus angle_col (which elevation angle the curve came from).
         Saved next to reps.parquet as rep_curves.parquet (save_curves / load_curves).
         Curves of different angles are never mixed: fitting or scoring raises.
         add_dtw_dist also checks the angle ACROSS tables: the curves, the
         training reps and the scored reps must all record the same valid
         angle_col (PR #15 review); missing or unknown provenance raises.
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
    """One normalised, stretched curve per rep (index rep_id, CURVE_COLUMNS + angle_col).
    Reps too short or too gappy to resample are left out (they get dtw_dist NaN)."""
    from neuroflow.segmentation.segment import reps_angle_col
    angle = reps_angle_col(reps, angle_col)
    R, kept = rep_matrix(frames, reps, rest_ref, rom_ref, n=TEMPLATE_LEN, angle_col=angle)
    if len(kept) == 0:
        R = np.empty((0, TEMPLATE_LEN))
    out = pd.DataFrame(R, index=pd.Index(kept, name="rep_id"), columns=CURVE_COLUMNS)
    out["angle_col"] = angle
    return out


def save_curves(curves, path):
    curves.reset_index().to_parquet(path, index=False)


def load_curves(path):
    c = pd.read_parquet(path)
    missing = [k for k in ["rep_id", "angle_col"] + CURVE_COLUMNS if k not in c.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing[:5]}")
    return c.set_index("rep_id")


def check_angles(curves, *rep_tables):
    """
    The one angle that the curves and every reps table share; raises otherwise.
    Each must record angle_col on every row, with one value from ANGLE_COLS, and
    all must agree, so a template built on one angle definition is never compared
    with reps or curves of another (PR #15 review).
    """
    from neuroflow.segmentation.segment import ANGLE_COLS

    def one(table, what):
        if "angle_col" not in table.columns:
            raise ValueError(f"{what} don't record angle_col; the angle can't be established")
        col = table["angle_col"]
        if col.isna().any():
            raise ValueError(f"{what}: {int(col.isna().sum())} rows without angle_col")
        vals = sorted(col.unique())
        if len(vals) > 1:
            raise ValueError(f"{what} mix angle columns {vals}; never compare them")
        if vals and vals[0] not in ANGLE_COLS:
            raise ValueError(f"{what}: angle_col {vals[0]!r} is not one of {ANGLE_COLS}")
        return vals[0] if vals else None

    angle = one(curves, "curves")
    for i, reps in enumerate(rep_tables):
        a = one(reps, "training reps" if i == 0 else "scored reps")
        if a is not None and a != angle:
            raise ValueError(f"curves are on {angle}, but the "
                             f"{'training' if i == 0 else 'scored'} reps are on {a}")
    return angle


def curves_angle(curves):
    """The one angle these curves come from; raises if they mix angles."""
    if "angle_col" not in curves.columns:
        return None                       # bare curve tables (tests, mocks)
    angles = sorted(curves["angle_col"].dropna().unique())
    if len(angles) > 1:
        raise ValueError(f"curves mix angle columns {angles}; never compare them")
    return angles[0] if angles else None


# ---- C's side: template and distance inside a fold ----

def fit_template(curves, train_reps, method="dba", band_frac=BAND_FRAC):
    """
    Template from the NORMAL reps of train_reps that have a curve.
    Pass the fold's training subjects only. method: "dba" (B5 default) or "mean".
    """
    curves_angle(curves)
    ids = train_reps.loc[train_reps["condition"] == "normal", "rep_id"]
    R = curves.reindex(ids)[CURVE_COLUMNS].dropna().to_numpy()
    if len(R) == 0:
        raise ValueError("no normal training reps with a curve to build the template from")
    if method == "mean":
        return mean_template(R)
    if method == "dba":
        return dba_template(R, band_frac=band_frac)[0]
    raise ValueError(f"method must be 'dba' or 'mean', got {method!r}")


def dtw_dist(curves, template, band_frac=BAND_FRAC):
    """Banded DTW distance of every curve to the template (Series by rep_id)."""
    d = [dtw_distance(x, template, band_frac) for x in curves[CURVE_COLUMNS].to_numpy()]
    return pd.Series(d, index=curves.index, name="dtw_dist", dtype=float)


def add_dtw_dist(reps, curves, train_reps, method="dba", band_frac=BAND_FRAC):
    """
    Return a copy of reps with a dtw_dist column, using a template fitted on the
    normal reps of train_reps (the fold's training subjects). reps may be the
    held-out subject's reps, or all reps; reps without a curve get NaN.
    Only curves of train_reps' normal reps enter the template, so the held-out
    subject never shapes its own reference.
    """
    check_angles(curves, train_reps, reps)
    template = fit_template(curves, train_reps, method, band_frac)
    have = curves.reindex(reps["rep_id"]).dropna(subset=CURVE_COLUMNS)
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
    reps = pd.DataFrame(rows).assign(angle_col="theta_ht_deg")
    curves = pd.DataFrame(curves, index=pd.Index(reps["rep_id"], name="rep_id"),
                          columns=CURVE_COLUMNS).assign(angle_col="theta_ht_deg")

    for held in ("sub0", "sub6"):
        train = reps[reps["subject_id"] != held]
        out = add_dtw_dist(reps[reps["subject_id"] == held], curves, train)
        print(f"held out {held}: median dtw_dist {out['dtw_dist'].median():.3f}")


if __name__ == "__main__":
    main()
