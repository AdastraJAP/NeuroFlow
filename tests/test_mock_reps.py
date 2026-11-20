"""C1 tests: mock reps generator."""
import numpy as np
import pandas as pd
import pytest

from fixtures.make_mock_reps import (BLOCKS, N_SUBJECTS, REP_COLUMNS, check,
                                     make_mock_reps)


@pytest.fixture(scope="module")
def reps():
    return make_mock_reps()


def test_passes_its_own_checks(reps):
    check(reps)


def test_shape_and_columns(reps):
    assert list(reps.columns) == REP_COLUMNS
    assert reps["subject_id"].nunique() == N_SUBJECTS
    assert reps["block_id"].nunique() == len(BLOCKS)
    assert (reps["schema_version"] == 1).all()


def test_same_seed_same_table():
    pd.testing.assert_frame_equal(make_mock_reps(seed=7, n_subjects=3),
                                  make_mock_reps(seed=7, n_subjects=3))


def test_different_seed_different_table():
    a = make_mock_reps(seed=1, n_subjects=3)
    b = make_mock_reps(seed=2, n_subjects=3)
    assert not np.allclose(a["dtw_dist"], b["dtw_dist"])


def test_slow_is_longer_for_every_subject(reps):
    m = reps.groupby(["subject_id", "condition", "level"])["duration_s"].median()
    ratio = m.xs(("slow", "2x"), level=("condition", "level")) / \
        m.xs(("normal", "none"), level=("condition", "level"))
    assert ratio.between(1.6, 2.4).all()


def test_subject_effect_exists(reps):
    # Subjects differ more than reps inside one subject: needed for P1 to show leakage
    normal = reps[reps["condition"] == "normal"]
    between = normal.groupby("subject_id")["rom_peak_deg"].mean().std()
    within = normal.groupby("subject_id")["rom_peak_deg"].std().mean()
    assert between > within


def test_compensation_features_move_the_right_way(reps):
    m = reps.groupby("condition")[["max_trunk_lean_deg", "min_elbow_deg",
                                   "max_hike_ratio"]].mean()
    assert m.loc["trunk_lean", "max_trunk_lean_deg"] > m.loc["normal", "max_trunk_lean_deg"] + 4
    assert m.loc["elbow_bend", "min_elbow_deg"] < m.loc["normal", "min_elbow_deg"] - 25
    assert m.loc["hike", "max_hike_ratio"] > m.loc["normal", "max_hike_ratio"]


def test_missing_values_are_nan_not_zero(reps):
    unknown = reps["t_start_ms"].isna()
    assert unknown.any()
    assert reps.loc[unknown, "duration_s"].isna().all()
    assert (reps["duration_s"].dropna() > 0).all()


def test_rater_labels_only_on_rated_conditions(reps):
    rated = reps.loc[reps["rater_labels"].notna(), "condition"].unique()
    assert set(rated) <= {"normal", "trunk_lean", "elbow_bend", "hike"}
