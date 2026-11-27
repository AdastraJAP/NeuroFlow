"""
Which reps are calibration, test or practice (owner: Wilkend).

One place for the rule, so the scorer, harness, statistics and P1 never look for
mock names (normal_calib / normal_test / s1 / s2) by themselves.

Every rep gets its role on its own row, so v1 and v2 reps can be mixed:

  1. calibration: an explicit boolean `is_calib` where it is set (e.g. from B's
     calibration manifest); where it is null, the role below decides;
  2. role: schema v2 `block_role` = calib | practice | test (issue #14) where it
     is set; where it is null (a v1 table, or v2 reps segmented from v1 frames,
     which B leaves null rather than guess), the v1 rule from block_id:
     "normal_calib" = calib, anything else = test;
  3. normal test reps: condition "normal" and role test; on rows that fell back
     to the v1 rule, the block must also be "normal_test".

A block_role outside calib | practice | test raises instead of silently
dropping the rep.

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


def _has_role(reps):
    """True where the row carries a v2 block_role (False on v1 rows)."""
    if "block_role" not in reps.columns:
        return pd.Series(False, index=reps.index)
    return reps["block_role"].notna()


def block_roles(reps):
    """Role of every rep: block_role where set, else the v1 rule from block_id."""
    has = _has_role(reps)
    bad = sorted(set(reps.loc[has, "block_role"]) - set(BLOCK_ROLES)) if has.any() else []
    if bad:
        raise ValueError(f"block_role must be one of {BLOCK_ROLES}, got {bad}")
    if "block_id" in reps.columns:
        legacy = np.where(reps["block_id"].eq(LEGACY_CALIB_BLOCK), CALIB_ROLE, TEST_ROLE)
    else:
        legacy = np.full(len(reps), TEST_ROLE, dtype=object)
    role = pd.Series(legacy, index=reps.index, dtype=object)
    if has.any():
        role[has] = reps.loc[has, "block_role"]
    return role


def calib_mask(reps):
    """True for calibration reps: is_calib where set, else role == calib."""
    m = block_roles(reps).eq(CALIB_ROLE)
    if "is_calib" in reps.columns:
        flag = reps["is_calib"].astype("boolean")
        m = flag.fillna(m).astype(bool)
    return m


def test_mask(reps):
    """True for reps that may be evaluated: test reps, never calibration or practice."""
    return block_roles(reps).eq(TEST_ROLE) & ~calib_mask(reps)


def normal_test_mask(reps):
    """Normal reps on the evaluation side (v1 rows: the normal_test block only)."""
    m = reps["condition"].eq("normal") & test_mask(reps)
    if "block_id" in reps.columns:
        m &= _has_role(reps) | reps["block_id"].eq(LEGACY_TEST_BLOCK)
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
