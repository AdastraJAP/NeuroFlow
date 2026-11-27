"""Per-fold dtw_dist: curves from frames (B), template + distance inside a fold (C)."""
import os

import numpy as np
import pandas as pd
import pytest

from neuroflow.segmentation.dtw import dtw_distance
from neuroflow.segmentation.dtw_feature import (
    CURVE_COLUMNS, add_dtw_dist, dtw_dist, fit_template, rep_curves)
from neuroflow.segmentation.segment import calibration_refs, segment_reps
from neuroflow.segmentation.templates import TEMPLATE_LEN, timing_varied_reps

FX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _toy():
    """5 normal subjects + 1 subject with half-height (capped) reps."""
    rows, X = [], []
    for s in range(5):
        for k, x in enumerate(timing_varied_reps(k=8, seed=s)):
            rows.append({"rep_id": f"s{s}_{k}", "subject_id": f"s{s}", "condition": "normal"})
            X.append(x)
    for k, x in enumerate(timing_varied_reps(k=8, seed=50)):
        rows.append({"rep_id": f"cap_{k}", "subject_id": "cap", "condition": "rom_cap"})
        X.append(0.5 * x)
    reps = pd.DataFrame(rows).assign(angle_col="theta_ht_deg")
    curves = pd.DataFrame(X, index=pd.Index(reps["rep_id"], name="rep_id"), columns=CURVE_COLUMNS)
    return reps, curves.assign(angle_col="theta_ht_deg")


def test_curves_from_mock_frames():
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    rest, rom = calibration_refs()[frames["subject_id"].iloc[0]]
    reps = segment_reps(frames, rom, rest)
    c = rep_curves(frames, reps, rest, rom)
    assert list(c.columns) == CURVE_COLUMNS + ["angle_col"] and len(CURVE_COLUMNS) == TEMPLATE_LEN
    assert list(c.index) == list(reps["rep_id"])
    # range-normalised: starts near rest (0) and peaks near the typical peak (1)
    assert np.abs(c["c00"]).max() < 0.2
    assert c[CURVE_COLUMNS].max(axis=1).between(0.8, 1.2).all()


def test_capped_reps_are_further_from_normal_template():
    reps, curves = _toy()
    train = reps[reps["subject_id"].isin(["s0", "s1", "s2", "s3"])]
    test = reps[reps["subject_id"].isin(["s4", "cap"])]
    out = add_dtw_dist(test, curves, train)
    d = out.groupby("subject_id")["dtw_dist"].median()
    assert d["cap"] > 3 * d["s4"]


def test_template_uses_normal_train_reps_only():
    reps, curves = _toy()
    train = reps[reps["subject_id"] != "s4"]                 # includes the capped subject
    t1 = fit_template(curves, train, method="mean")
    t2 = fit_template(curves, train[train["condition"] == "normal"], method="mean")
    np.testing.assert_array_equal(t1, t2)                    # capped reps never enter it
    # the held-out subject's curves can change freely without changing the template
    other = curves.copy()
    other.loc[other.index.str.startswith("s4_"), CURVE_COLUMNS] *= 10
    np.testing.assert_array_equal(t1, fit_template(other, train, method="mean"))


def test_distance_matches_b4_and_missing_curve_is_nan():
    reps, curves = _toy()
    train = reps[reps["subject_id"] != "s0"]
    t = fit_template(curves, train)
    d = dtw_dist(curves.loc[["s0_0"]], t)
    assert d.iloc[0] == pytest.approx(dtw_distance(curves.loc["s0_0", CURVE_COLUMNS].to_numpy(dtype=float), t, 0.15))
    out = add_dtw_dist(reps[reps["subject_id"] == "s0"], curves.drop(index="s0_1"), train)
    assert np.isnan(out.set_index("rep_id").loc["s0_1", "dtw_dist"])
    assert out["dtw_dist"].notna().sum() == 7


def test_no_normal_training_reps_raises():
    reps, curves = _toy()
    with pytest.raises(ValueError, match="no normal training reps"):
        fit_template(curves, reps[reps["condition"] != "normal"])


# ---- angle provenance and saving (#14) ----
from neuroflow.segmentation.dtw_feature import load_curves, save_curves  # noqa: E402


def test_curves_record_their_angle_and_round_trip(tmp_path):
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    rest, rom = calibration_refs()[frames["subject_id"].iloc[0]]
    c = rep_curves(frames, segment_reps(frames, rom, rest), rest, rom)
    assert (c["angle_col"] == "theta_ht_deg").all()
    p = tmp_path / "rep_curves.parquet"
    save_curves(c, p)
    back = load_curves(p)
    pd.testing.assert_frame_equal(back, c)


def test_mixed_angles_are_refused():
    reps, curves = _toy()
    curves.iloc[0, curves.columns.get_loc("angle_col")] = "theta_v_deg"
    with pytest.raises(ValueError, match="mix angle columns"):
        add_dtw_dist(reps, curves, reps[reps["subject_id"] != "s0"])


# ---- angle provenance across tables (PR #15 review) ----

def test_uniform_angle_mismatch_across_tables_is_refused():
    """Curves all on theta_ht_deg, reps all on theta_v_deg: each table is uniform on
    its own, but they must never be compared."""
    reps, curves = _toy()
    train = reps[reps["subject_id"] != "s0"]
    target = reps[reps["subject_id"] == "s0"]
    v = lambda r: r.assign(angle_col="theta_v_deg")          # noqa: E731
    with pytest.raises(ValueError, match="training reps are on theta_v_deg"):
        add_dtw_dist(v(target), curves, v(train))
    with pytest.raises(ValueError, match="scored reps are on theta_v_deg"):
        add_dtw_dist(v(target), curves, train)
    with pytest.raises(ValueError, match="training reps are on theta_v_deg"):
        add_dtw_dist(target, curves, v(train))
    assert add_dtw_dist(target, curves, train)["dtw_dist"].notna().all()   # all agree: fine


@pytest.mark.parametrize("break_it", ["curves_no_column", "curves_null", "reps_no_column",
                                      "reps_null", "unknown_angle"])
def test_missing_or_invalid_angle_provenance_is_refused(break_it):
    reps, curves = _toy()
    train = reps[reps["subject_id"] != "s0"]
    target = reps[reps["subject_id"] == "s0"].copy()
    if break_it == "curves_no_column":
        curves = curves.drop(columns="angle_col")
    elif break_it == "curves_null":
        curves.iloc[0, curves.columns.get_loc("angle_col")] = None
    elif break_it == "reps_no_column":
        target = target.drop(columns="angle_col")
    elif break_it == "reps_null":
        target.iloc[0, target.columns.get_loc("angle_col")] = None
    else:
        curves = curves.assign(angle_col="theta_xyz")
    with pytest.raises(ValueError):
        add_dtw_dist(target, curves, train)
