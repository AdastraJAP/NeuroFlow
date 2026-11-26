"""C7 v0 tests: tempo term x mapping grid."""
import numpy as np
import pytest

from fixtures.make_mock_reps import make_mock_reps
from neuroflow.scoring.ablation import (
    FULL_GRID, MAPPINGS, TEMPO_TERMS, make_cell, run_grid,
)
from neuroflow.scoring.folds import make_folds
from neuroflow.scoring.harness import auroc_by_condition, run_loso


@pytest.fixture(scope="module")
def reps():
    return make_mock_reps()


@pytest.fixture(scope="module")
def folds(reps):
    return make_folds(reps["subject_id"])


def _slow_auroc(reps, folds, tempo, mapping):
    fit, score = make_cell(tempo, mapping)
    t = auroc_by_condition(run_loso(reps, folds, fit, score), reps)
    return t.query("condition == 'slow' and level == '2x'")["auroc"].item()


def test_tempo_term_fixes_slow_reps(reps, folds):
    # The P2 claim on the mock: shape alone is blind to slow reps, a tempo term is not.
    assert _slow_auroc(reps, folds, "none", "lw_ecdf") < 0.65
    assert _slow_auroc(reps, folds, "duration", "lw_ecdf") > 0.95


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_every_mapping_scores_in_range(reps, folds, mapping):
    fit, score = make_cell("duration", mapping)
    s = run_loso(reps, folds, fit, score)
    ok = s.loc[~s["abstain"], "score_0_100"]
    assert ok.between(0, 100).all()
    assert s["abstain"].mean() < 0.02


def test_cells_refuse_training_subjects(reps):
    fit, score = make_cell("duration+ttp", "gmm")
    model = fit(reps[reps["subject_id"] != "sub01"])
    with pytest.raises(ValueError, match="fitted on"):
        score(model, reps[reps["subject_id"] == "sub02"], fold_id=-1)


def test_fit_ignores_impaired_training_reps(reps):
    fit, _ = make_cell("duration", "lw_ecdf")
    train = reps[reps["subject_id"] != "sub01"]
    other = train.copy()
    other.loc[other["condition"] != "normal", "dtw_dist"] *= 10
    np.testing.assert_allclose(fit(train)["ref"], fit(other)["ref"])


def test_unknown_mapping_raises():
    with pytest.raises(ValueError):
        make_cell("none", "nope")


@pytest.mark.slow
def test_grid_shape_and_cis(reps, folds):
    grid = run_grid(reps, folds, n_boot=50)
    assert len(grid) == len(TEMPO_TERMS) * len(MAPPINGS) * 11
    in_full = grid.apply(lambda r: (r["condition"], r["level"]) in FULL_GRID, axis=1)
    assert grid.loc[in_full, ["lo", "hi"]].notna().all().all()
    assert grid.loc[~in_full, ["lo", "hi"]].isna().all().all()
    assert (grid.loc[in_full, "lo"] <= grid.loc[in_full, "auroc"] + 1e-12).all()
