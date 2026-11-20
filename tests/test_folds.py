"""C1 tests: subject-wise folds."""
import pandas as pd
import pytest

from neuroflow.scoring.folds import (check_folds, iter_folds, load_folds,
                                     make_folds, save_folds)

SUBJECTS = [f"sub{i:02d}" for i in range(1, 19)]


def toy_reps(subjects=SUBJECTS, per_subject=3):
    rows = [{"rep_id": f"{s}_{k}", "subject_id": s} for s in subjects for k in range(per_subject)]
    return pd.DataFrame(rows)


def test_every_subject_once():
    folds = make_folds(SUBJECTS)
    check_folds(folds)
    assert sorted(folds["subject_id"]) == SUBJECTS


def test_duplicates_in_input_are_fine():
    folds = make_folds(SUBJECTS + SUBJECTS)
    assert len(folds) == len(SUBJECTS)


def test_locked_count_and_loso():
    folds = make_folds(SUBJECTS, n_locked=4)
    assert (folds["split"] == "locked_test").sum() == 4
    dev = folds[folds["split"] == "train"]
    assert dev["fold_id"].nunique() == 14          # LOSO: one subject per fold


def test_k_folds():
    folds = make_folds(SUBJECTS, n_locked=4, n_folds=7)
    dev = folds[folds["split"] == "train"]
    assert dev["fold_id"].nunique() == 7
    assert (dev.groupby("fold_id").size() == 2).all()


def test_same_seed_same_folds():
    pd.testing.assert_frame_equal(make_folds(SUBJECTS, seed=3), make_folds(SUBJECTS, seed=3))


def test_iter_folds_never_mixes_subjects():
    reps, folds = toy_reps(), make_folds(SUBJECTS)
    locked = set(folds.loc[folds["split"] == "locked_test", "subject_id"])
    n = 0
    for _, train, test in iter_folds(reps, folds):
        n += 1
        assert not set(train["subject_id"]) & set(test["subject_id"])
        assert not (set(train["subject_id"]) | set(test["subject_id"])) & locked
        assert len(train) + len(test) == 3 * (len(SUBJECTS) - len(locked))
    assert n == len(SUBJECTS) - len(locked)


def test_every_dev_subject_is_tested_exactly_once():
    reps, folds = toy_reps(), make_folds(SUBJECTS)
    tested = [s for _, _, test in iter_folds(reps, folds) for s in test["subject_id"].unique()]
    assert len(tested) == len(set(tested)) == len(SUBJECTS) - 4


def test_unknown_subject_is_an_error():
    folds = make_folds(SUBJECTS)
    with pytest.raises(ValueError):
        list(iter_folds(toy_reps(SUBJECTS + ["stranger"]), folds))


def test_check_catches_a_subject_in_two_folds():
    folds = make_folds(SUBJECTS)
    bad = pd.concat([folds, folds.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError):
        check_folds(bad)


def test_bad_arguments():
    with pytest.raises(ValueError):
        make_folds(SUBJECTS[:3], n_locked=3)
    with pytest.raises(ValueError):
        make_folds(SUBJECTS, n_folds=1)


def test_round_trip(tmp_path):
    folds = make_folds(SUBJECTS)
    path = tmp_path / "folds.csv"
    save_folds(folds, path)
    pd.testing.assert_frame_equal(load_folds(path), folds)
