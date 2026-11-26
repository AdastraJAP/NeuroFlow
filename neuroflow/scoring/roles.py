"""
Which reps are calibration, test or practice (owner: Wilkend).

One place for the rule, so the scorer, harness, statistics and P1 never look for
mock names (normal_calib / normal_test / s1 / s2) by themselves. Precedence:

  1. an explicit boolean column `is_calib` (e.g. set from B's calibration
     manifest, when the calibration reps were recorded as practice clips);
  2. schema v2 `block_role` = calib | practice | test  (issue #14);
  3. schema v1 / the mock: block_id == "normal_calib" (calibration) and
     block_id == "normal_test" (normal test reps).

Baseline session (issue #14 / PR #8 review): the personal duration median comes
from each subject's calibration reps of its BASELINE session = the first
session_id in sorted order. session_id may be null (G1 practice clips); a subject
whose calibration reps have no session id uses all of them.
"""

import numpy as np
import pandas as pd

CALIB_ROLE = "calib"
TEST_ROLE = "test"
PRACTICE_ROLE = "practice"
BLOCK_ROLES = (CALIB_ROLE, PRACTICE_ROLE, TEST_ROLE)

LEGACY_CALIB_BLOCK = "normal_calib"
LEGACY_TEST_BLOCK = "normal_test"

# Columns the role rules may need when scores are joined back to reps.
ROLE_COLUMNS = ("block_id", "block_role", "is_calib", "session_id", "condition", "level")


def role_columns(reps):
    """The ROLE_COLUMNS that this reps table actually has."""
    return [c for c in ROLE_COLUMNS if c in reps.columns]


def calib_mask(reps):
    """True for calibration reps."""
    if "is_calib" in reps.columns:
        return reps["is_calib"].astype("boolean").fillna(False).astype(bool)
    if "block_role" in reps.columns:
        return reps["block_role"].eq(CALIB_ROLE)
    return reps["block_id"].eq(LEGACY_CALIB_BLOCK)


def test_mask(reps):
    """True for reps that may be evaluated: test reps, never calibration or practice."""
    if "block_role" in reps.columns:
        return reps["block_role"].eq(TEST_ROLE) & ~calib_mask(reps)
    return ~calib_mask(reps)


def normal_test_mask(reps):
    """Normal reps on the evaluation side (v1: the normal_test block)."""
    m = reps["condition"].eq("normal") & test_mask(reps)
    if "block_role" not in reps.columns:
        m &= reps["block_id"].eq(LEGACY_TEST_BLOCK)
    return m


def session_mask(reps, session):
    """True for every rep when session is None, else for that session only."""
    if session is None:
        return pd.Series(True, index=reps.index)
    return reps["session_id"].eq(session)


def baseline_calib_mask(reps):
    """Calibration reps of each subject's baseline session (see module docstring)."""
    m = calib_mask(reps)
    keep = pd.Series(False, index=reps.index)
    for _, g in reps[m].groupby("subject_id", sort=False):
        sessions = g["session_id"].dropna()
        if len(sessions):
            first = sorted(sessions.unique())[0]
            keep.loc[g.index[g["session_id"].eq(first).to_numpy()]] = True
        else:
            keep.loc[g.index] = True
    return keep


def calib_duration_median(reps):
    """Median calibration duration per subject (baseline session); subjects without one are absent."""
    calib = reps[baseline_calib_mask(reps)]
    return calib.groupby("subject_id")["duration_s"].median().dropna()
