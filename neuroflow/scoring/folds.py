"""
C1: subject-wise folds (owner: Member C).

Why by subject: if one person's reps sit in both the training set and the test
set, the scorer has already seen how that person moves, and the result looks
better than it will for a new user. That optimism is exactly what P1 measures,
so the folds must never let it in by accident.

folds.csv, one row per subject:
    subject_id
    fold_id    which fold the subject is held out in; -1 for the locked test
    split      "train" (development pool, used for LOSO) or "locked_test"

Locked test: a few subjects are set aside and never used for fitting or tuning.
They are scored once, at gate G5, by a member who did not tune on them.

Default is leave-one-subject-out (LOSO): every development subject is its own
fold. n_folds=k groups the development subjects into k folds instead.

Run (from the repo root):
    python -m neuroflow.scoring.folds
"""

import os

import numpy as np
import pandas as pd

from neuroflow.scoring.contract import FOLD_COLUMNS, LOCKED_FOLD_ID, SPLITS

SEED = 0
N_LOCKED = 4
OUT_PATH = os.path.join("fixtures", "mock_folds.csv")


def make_folds(subject_ids, n_locked=N_LOCKED, n_folds=None, seed=SEED):
    """
    Assign every subject to one fold.
    subject_ids: any iterable (duplicates are fine, e.g. a column of the reps table).
    Returns a DataFrame with FOLD_COLUMNS, one row per subject.
    """
    subjects = sorted(set(subject_ids))
    if n_locked >= len(subjects):
        raise ValueError("n_locked must leave at least one development subject")
    rng = np.random.default_rng(seed)
    shuffled = [subjects[i] for i in rng.permutation(len(subjects))]

    locked, dev = shuffled[:n_locked], shuffled[n_locked:]
    k = len(dev) if n_folds is None else n_folds
    if not 2 <= k <= len(dev):
        raise ValueError(f"n_folds must be between 2 and {len(dev)}")

    rows = [(s, LOCKED_FOLD_ID, "locked_test") for s in locked]
    rows += [(s, i % k, "train") for i, s in enumerate(dev)]
    folds = pd.DataFrame(rows, columns=FOLD_COLUMNS)
    return folds.sort_values("subject_id").reset_index(drop=True)


def check_folds(folds):
    """Raise AssertionError if the folds break a rule."""
    assert list(folds.columns) == FOLD_COLUMNS
    assert folds["subject_id"].is_unique, "a subject appears in two folds"
    assert folds["split"].isin(SPLITS).all()
    locked = folds["split"] == "locked_test"
    assert (folds.loc[locked, "fold_id"] == LOCKED_FOLD_ID).all()
    assert (folds.loc[~locked, "fold_id"] >= 0).all()


def iter_folds(reps, folds):
    """
    Yield (fold_id, train_reps, test_reps) for every development fold.
    Locked-test subjects are in neither part. Train and test never share a subject.
    """
    check_folds(folds)
    dev = folds[folds["split"] == "train"]
    fold_of = dict(zip(dev["subject_id"], dev["fold_id"]))
    unknown = set(reps["subject_id"]) - set(folds["subject_id"])
    if unknown:
        raise ValueError(f"subjects missing from folds: {sorted(unknown)}")

    in_dev = reps["subject_id"].isin(fold_of)
    rep_fold = reps["subject_id"].map(fold_of)
    for fold_id in sorted(dev["fold_id"].unique()):
        test = reps[in_dev & (rep_fold == fold_id)]
        train = reps[in_dev & (rep_fold != fold_id)]
        assert not set(train["subject_id"]) & set(test["subject_id"])
        yield int(fold_id), train, test


def save_folds(folds, path):
    check_folds(folds)
    folds.to_csv(path, index=False)


def load_folds(path):
    folds = pd.read_csv(path, dtype={"subject_id": str})
    check_folds(folds)
    return folds


def main():
    from fixtures.make_mock_reps import make_mock_reps

    print(f"seed = {SEED}")
    reps = make_mock_reps()
    folds = make_folds(reps["subject_id"])
    check_folds(folds)

    locked = folds.loc[folds["split"] == "locked_test", "subject_id"].tolist()
    print(f"{len(folds)} subjects: {len(folds) - len(locked)} development (LOSO), "
          f"{len(locked)} locked test {locked}")

    n, seen = 0, set()
    for fold_id, train, test in iter_folds(reps, folds):
        n += 1
        seen |= set(train["subject_id"]) | set(test["subject_id"])
        if fold_id < 3:
            print(f"  fold {fold_id}: test {sorted(test['subject_id'].unique())}, "
                  f"train {train['subject_id'].nunique()} subjects, "
                  f"{len(train)} train reps / {len(test)} test reps")
    assert not seen & set(locked), "a locked-test subject leaked into a fold"
    print(f"  ... {n} folds; no subject in two parts of any fold; "
          f"locked subjects never used")

    save_folds(folds, OUT_PATH)
    pd.testing.assert_frame_equal(load_folds(OUT_PATH), folds)
    print(f"saved {OUT_PATH} (round trip OK)")


if __name__ == "__main__":
    main()
