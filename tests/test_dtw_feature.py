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
    reps = pd.DataFrame(rows)
    curves = pd.DataFrame(X, index=pd.Index(reps["rep_id"], name="rep_id"), columns=CURVE_COLUMNS)
    return reps, curves


def test_curves_from_mock_frames():
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    rest, rom = calibration_refs()[frames["subject_id"].iloc[0]]
    reps = segment_reps(frames, rom, rest)
    c = rep_curves(frames, reps, rest, rom)
    assert list(c.columns) == CURVE_COLUMNS and len(c.columns) == TEMPLATE_LEN
    assert list(c.index) == list(reps["rep_id"])
    # range-normalised: starts near rest (0) and peaks near the typical peak (1)
    assert np.abs(c.iloc[:, 0]).max() < 0.2
    assert c.max(axis=1).between(0.8, 1.2).all()


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
    other.loc[other.index.str.startswith("s4_")] *= 10
    np.testing.assert_array_equal(t1, fit_template(other, train, method="mean"))


def test_distance_matches_b4_and_missing_curve_is_nan():
    reps, curves = _toy()
    train = reps[reps["subject_id"] != "s0"]
    t = fit_template(curves, train)
    d = dtw_dist(curves.loc[["s0_0"]], t)
    assert d.iloc[0] == pytest.approx(dtw_distance(curves.loc["s0_0"].to_numpy(), t, 0.15))
    out = add_dtw_dist(reps[reps["subject_id"] == "s0"], curves.drop(index="s0_1"), train)
    assert np.isnan(out.set_index("rep_id").loc["s0_1", "dtw_dist"])
    assert out["dtw_dist"].notna().sum() == 7


def test_no_normal_training_reps_raises():
    reps, curves = _toy()
    with pytest.raises(ValueError, match="no normal training reps"):
        fit_template(curves, reps[reps["condition"] != "normal"])
