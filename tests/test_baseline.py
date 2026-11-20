"""C2 tests: RehabHub baseline scorer."""
import numpy as np
import pandas as pd
import pytest

from neuroflow.scoring.baseline import (fit_baseline, rehabhub_score,
                                        score_baseline, z_score)
from neuroflow.scoring.contract import SCORE_COLUMNS


def toy(dtw, condition="normal", subject="a"):
    return pd.DataFrame({
        "rep_id": [f"{subject}_{i}" for i in range(len(dtw))],
        "subject_id": subject,
        "condition": condition,
        "dtw_dist": dtw,
    })


def test_hand_worked_example():
    # normal distances 1, 2, 3: mu = 2, sigma = 1 (sample SD)
    model = fit_baseline(toy([1.0, 2.0, 3.0]))
    assert model["mu"] == pytest.approx(2.0)
    assert model["sigma"] == pytest.approx(1.0)
    assert z_score(model, [2.0, 4.0, 6.0]) == pytest.approx([0.0, 2.0, 4.0])


def test_score_formula():
    z = [-3.0, 0.0, 0.99, 1.0, 2.0, 4.0]
    assert rehabhub_score(z) == pytest.approx([100, 100, 100, 100, 50, 25])


def test_saturation():
    # everything within one SD above the mean scores exactly 100
    model = fit_baseline(toy([1.0, 2.0, 3.0]))
    scores = score_baseline(model, toy([0.5, 1.0, 2.0, 2.9, 3.0], subject="b"))
    assert (scores["score_0_100"] == 100).all()


def test_fit_ignores_impaired_reps():
    normal = toy([1.0, 2.0, 3.0])
    both = pd.concat([normal, toy([50.0, 80.0], condition="slow")], ignore_index=True)
    assert fit_baseline(both)["mu"] == fit_baseline(normal)["mu"]


def test_missing_distance_abstains():
    model = fit_baseline(toy([1.0, 2.0, 3.0]))
    scores = score_baseline(model, toy([np.nan, 2.0], subject="b"))
    assert np.isnan(scores["score_0_100"].iloc[0])
    assert scores["abstain"].tolist() == [True, False]
    assert scores["abstain_reason"].iloc[0] == "no_dtw_distance"


def test_output_matches_contract():
    model = fit_baseline(toy([1.0, 2.0, 3.0]))
    scores = score_baseline(model, toy([1.0], subject="b"), fold_id=3)
    assert list(scores.columns) == SCORE_COLUMNS
    assert scores["fold_id"].iloc[0] == 3


def test_refuses_to_score_training_subjects_inside_a_fold():
    model = fit_baseline(toy([1.0, 2.0, 3.0], subject="a"))
    with pytest.raises(ValueError):
        score_baseline(model, toy([1.0], subject="a"), fold_id=0)


def test_needs_enough_normal_reps():
    with pytest.raises(ValueError):
        fit_baseline(toy([1.0]))
    with pytest.raises(ValueError):
        fit_baseline(toy([2.0, 2.0]))
