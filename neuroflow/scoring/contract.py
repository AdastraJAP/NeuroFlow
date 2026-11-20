"""
Column lists for the two tables Member C owns (schema_version 1).

    scores.parquet   one row per rep x scorer
    folds.csv        one row per subject

Change these only through a pull request that bumps SCHEMA_VERSION and tells
all three members (plan, section 2, rule 4).
"""

SCHEMA_VERSION = 1

SCORE_COLUMNS = [
    "schema_version", "rep_id", "scorer_id", "ref_type", "mapping", "fold_id",
    "score_0_100", "percentile", "subscores", "abstain", "abstain_reason",
    "feedback_code",
]

FOLD_COLUMNS = ["subject_id", "fold_id", "split"]

SPLITS = ("train", "val", "locked_test")
LOCKED_FOLD_ID = -1
