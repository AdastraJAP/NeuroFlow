"""
Scorer on schema-v2, G1-like reps (issue #14): block_role, real ids, null
calibration session, unavailable features, censored reps, angle provenance and
per-fold DTW templates. Built from the mock, so it runs without private data.
"""
import json

import numpy as np
import pandas as pd
import pytest

from fixtures.make_mock_reps import make_mock_reps
from neuroflow.scoring.harness import auroc_by_condition, run_loso
from neuroflow.scoring.scorer import fit_scorer, score_reps
from neuroflow.segmentation.dtw_feature import CURVE_COLUMNS

SUBJECTS = [f"sub{i:02d}" for i in range(1, 7)]
BLOCKS = ("normal_calib", "normal_test", "slow", "trunk_lean")


def _folds(reps):
    subj = sorted(reps["subject_id"].unique())
    return pd.DataFrame({"subject_id": subj, "fold_id": range(len(subj)), "split": "train"})


@pytest.fixture(scope="module")
def v2():
    """6 subjects, 5 reps per block and session, converted to schema v2 like G1."""
    r = make_mock_reps()
    r = r[r["subject_id"].isin(SUBJECTS)
          & r["block_id"].str.startswith(BLOCKS)
          & ~r["level"].isin(["3x", "small"])]
    r = r.groupby(["subject_id", "session_id", "block_id"], group_keys=False).head(5).copy()
    r["block_role"] = np.where(r["block_id"] == "normal_calib", "calib", "test")
    r["block_id"] = "T_" + r["block_id"]                        # names mean nothing now
    r["session_id"] = r["session_id"].map({"s1": "S001", "s2": "S002"})
    r["angle_col"] = "theta_v_deg"
    r["min_elbow_deg"] = np.nan                                 # hip/wrist-free G1 clips
    r["max_trunk_lean_deg"] = np.nan
    return r.reset_index(drop=True)


def test_scorer_runs_on_v2_and_drops_unavailable_features(v2):
    model = fit_scorer(v2[v2["subject_id"] != "sub01"])
    assert set(model["dropped_features"]) == {"min_elbow_deg", "max_trunk_lean_deg"}
    assert "duration_ratio" in model["features"]
    assert model["angle_col"] == "theta_v_deg"
    out = score_reps(model, v2[v2["subject_id"] == "sub01"], fold_id=0)
    ok = out[~out["abstain"]]
    assert len(ok) > 0
    assert "elbow" not in json.loads(ok["subscores"].iloc[0])  # sub-score left out too


def test_harness_uses_block_role_not_block_names(v2):
    scores = run_loso(v2, _folds(v2), fit_scorer, score_reps)
    t = auroc_by_condition(scores, v2, session=None)          # all sessions
    slow = t.query("condition == 'slow'")
    assert slow["n_normal"].item() > 0 and slow["auroc"].item() > 0.9


def test_calibration_from_practice_clips_with_null_session(v2):
    # G1: calibration reps came as practice clips with no session id; the
    # calibration manifest marks them, here via is_calib.
    r = v2.copy()
    calib = r["block_role"] == "calib"
    r.loc[calib, "block_role"] = "practice"
    r.loc[calib, "session_id"] = None
    r["is_calib"] = calib
    model = fit_scorer(r[r["subject_id"] != "sub01"])
    out = score_reps(model, r[r["subject_id"] == "sub01"], fold_id=0)
    assert (~out["abstain"]).sum() > 0


def test_missing_calibration_abstains_with_reason_when_scoring(v2):
    model = fit_scorer(v2[v2["subject_id"] != "sub01"])
    new = v2[(v2["subject_id"] == "sub01") & (v2["block_role"] != "calib")]
    out = score_reps(model, new, fold_id=0)                    # no raise
    assert out["abstain"].all()
    assert (out["abstain_reason"] == "no_calibration").all()


def test_missing_calibration_still_fails_loudly_when_fitting(v2):
    train = v2[~((v2["subject_id"] == "sub02") & (v2["block_role"] == "calib"))]
    with pytest.raises(ValueError, match="sub02"):
        fit_scorer(train)


def test_censored_end_is_named(v2):
    r = v2.copy()
    i = r.index[(r["subject_id"] == "sub01") & (r["block_role"] == "test")][:3]
    r.loc[i, ["t_end_ms", "duration_s", "time_to_peak_ratio"]] = np.nan
    model = fit_scorer(r[r["subject_id"] != "sub01"])
    out = score_reps(model, r[r["subject_id"] == "sub01"], fold_id=0).set_index("rep_id")
    assert (out.loc[r.loc[i, "rep_id"], "abstain_reason"] == "censored_end").all()


def test_reason_counts_in_auroc_table(v2):
    r = v2.copy()
    i = r.index[(r["condition"] == "slow") & r["t_start_ms"].notna()][:4]
    r.loc[i, ["t_end_ms", "duration_s"]] = np.nan
    t = auroc_by_condition(run_loso(r, _folds(r), fit_scorer, score_reps), r, session=None)
    assert "censored_end: 4" in t.query("condition == 'slow'")["abstain_reasons"].item()


def test_one_model_never_mixes_angle_definitions(v2):
    r = v2.copy()
    r.loc[r["subject_id"] == "sub02", "angle_col"] = "theta_ht_deg"
    with pytest.raises(ValueError, match="angle"):
        fit_scorer(r[r["subject_id"] != "sub01"])
    model = fit_scorer(v2[v2["subject_id"] != "sub01"])
    other = v2[v2["subject_id"] == "sub01"].assign(angle_col="theta_ht_deg")
    out = score_reps(model, other, fold_id=0)
    assert (out["abstain_reason"] == "mixed_angle_col").all()


# ---- per-fold DTW template (B's add_dtw_dist) ----

def _curves(reps, seed=0, bump=None):
    """Range-normalised 60-sample curves; slow reps have the same shape (DTW is speed-blind)."""
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, len(CURVE_COLUMNS))
    base = np.sin(np.pi * u) ** 2
    rows = [(1 + r.rom_rel_personal) * base + rng.normal(0, 0.03, len(u))
            for r in reps.itertuples()]
    c = pd.DataFrame(rows, index=pd.Index(reps["rep_id"], name="rep_id"), columns=CURVE_COLUMNS)
    if bump is not None:
        c.loc[bump] = c.loc[bump] + 0.8 * np.sin(3 * np.pi * u)      # very different shapes
    return c


def test_per_fold_template_never_contains_the_held_out_subject(v2):
    r = v2.drop(columns=["dtw_dist"])
    folds = _folds(r)
    a_normal = r.loc[(r["subject_id"] == "sub01") & (r["condition"] == "normal"), "rep_id"]
    c1, c2 = _curves(r), _curves(r, bump=a_normal)    # differ ONLY in sub01's normal curves
    s1 = run_loso(r, folds, fit_scorer, score_reps, curves=c1, dtw_method="mean")
    s2 = run_loso(r, folds, fit_scorer, score_reps, curves=c2, dtw_method="mean")
    slow_a = set(r.loc[(r["subject_id"] == "sub01") & (r["condition"] == "slow"), "rep_id"])
    a1 = s1[s1["rep_id"].isin(slow_a)].set_index("rep_id")["score_0_100"]
    a2 = s2[s2["rep_id"].isin(slow_a)].set_index("rep_id")["score_0_100"]
    # sub01's own normal curves changed, but in sub01's fold the template comes
    # from the other subjects only, so sub01's (unchanged) slow reps score the same.
    pd.testing.assert_series_equal(a1.sort_index(), a2.sort_index())
    assert s1["abstain"].mean() < 0.05


def test_dtw_without_curves_or_column_is_simply_unavailable(v2):
    model = fit_scorer(v2.drop(columns=["dtw_dist"])[v2["subject_id"] != "sub01"])
    assert "dtw_dist" in model["dropped_features"]


def test_one_model_never_mixes_left_and_right(v2):
    # PR #7 review: left-arm reps must never enter a right-arm reference.
    r = v2.copy()
    r.loc[r["subject_id"] == "sub02", "side"] = "left"
    with pytest.raises(ValueError, match="side"):
        fit_scorer(r[r["subject_id"] != "sub01"])
    model = fit_scorer(v2[v2["subject_id"] != "sub01"])
    left = v2[v2["subject_id"] == "sub01"].assign(side="left")
    out = score_reps(model, left, fold_id=0)
    assert (out["abstain_reason"] == "mixed_side").all()


def test_every_abstain_reason_is_allowed(v2):
    from neuroflow.scoring.contract import is_abstain_reason
    r = v2.copy()
    r.loc[r.index[:3], ["t_end_ms", "duration_s"]] = np.nan
    out = run_loso(r, _folds(r), fit_scorer, score_reps)
    reasons = out.loc[out["abstain"], "abstain_reason"]
    assert len(reasons) > 0 and reasons.map(is_abstain_reason).all()


def test_reps_segmented_from_v1_frames_keep_their_roles(v2):
    # B's v2 segmenter leaves block_role null when the frames are v1 (mock, G1 export):
    # the v1 block names must still decide, per row
    r = v2.copy()
    r["block_id"] = r["block_id"].str.removeprefix("T_")
    r["block_role"] = None
    model = fit_scorer(r[r["subject_id"] != "sub01"])          # finds the calibration
    t = auroc_by_condition(run_loso(r, _folds(r), fit_scorer, score_reps), r, session=None)
    assert t.query("condition == 'slow'")["n_normal"].item() > 0
    assert "duration_ratio" in model["features"]
