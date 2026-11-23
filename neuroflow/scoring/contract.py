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

# Keys allowed in the scores.parquet "subscores" JSON string (None when abstained).
# "d" = the scorer's raw distance (C3), kept because score_0_100 floors at 0.
# Adding a key is not a column change, so SCHEMA_VERSION stays 1, but list it here.
SUBSCORE_KEYS = ("d", "rom", "tempo", "shape", "smooth", "trunk", "elbow", "hike")

# Values allowed in the "feedback_code" column (None = no feedback). Not "d" (it is
# a distance, not a sub-score) and not "hike" (exploratory: F1 ~ 0.55, too many
# false messages to show to a patient).
FEEDBACK_CODES = ("rom", "tempo", "shape", "smooth", "trunk", "elbow")

FOLD_COLUMNS = ["subject_id", "fold_id", "split"]

SPLITS = ("train", "val", "locked_test")
LOCKED_FOLD_ID = -1

# ref_type of rows scored on subjects the model was fitted on (P1's leaky
# within-participant split, the leak-test control). Only possible with an explicit
# allow_in_sample=True; fold_id never switches the leakage check off.
IN_SAMPLE_REF_TYPE = "in_sample"
