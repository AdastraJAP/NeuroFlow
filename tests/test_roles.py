"""
Role rules (neuroflow.scoring.roles) are resolved per row: block_role / is_calib
where they are set, the v1 block_id rule where they are null. B's v2 segmenter
writes block_role = null for blocks cut from v1 frames, so a null role must never
turn a calibration or test rep into "nothing" (PR #16 review).
"""
import numpy as np
import pandas as pd
import pytest

from neuroflow.scoring.contract import is_abstain_reason
from neuroflow.scoring import roles   # module import: pytest would collect roles.test_mask
from neuroflow.scoring.roles import block_roles, calib_duration_median, calib_mask, normal_test_mask

V1 = pd.DataFrame({
    "rep_id": ["a", "b", "c"],
    "subject_id": "sub01",
    "session_id": "s1",
    "block_id": ["normal_calib", "normal_test", "slow_2x"],
    "condition": ["normal", "normal", "slow"],
    "duration_s": [2.0, 2.1, 4.0],
})
V1_CALIB, V1_TEST, V1_NORMAL_TEST = [True, False, False], [False, True, True], [False, True, False]


def _check(reps, calib, test, normal_test):
    assert calib_mask(reps).tolist() == calib
    assert roles.test_mask(reps).tolist() == test
    assert normal_test_mask(reps).tolist() == normal_test


def test_v1_table_without_block_role():
    _check(V1, V1_CALIB, V1_TEST, V1_NORMAL_TEST)


@pytest.mark.parametrize("null", [None, np.nan])
def test_null_block_role_falls_back_to_block_id(null):
    # v2 reps from v1 frames: the column exists but has no value
    _check(V1.assign(block_role=null), V1_CALIB, V1_TEST, V1_NORMAL_TEST)
    assert calib_duration_median(V1.assign(block_role=null)).to_dict() == {"sub01": 2.0}


def test_block_role_wins_where_set():
    r = V1.assign(block_id=["T_01", "T_02", "T_03"],
                  block_role=["calib", "test", "practice"])
    _check(r, [True, False, False], [False, True, False], [False, True, False])


def test_mixed_v1_and_v2_rows():
    r = V1.assign(block_role=[None, "test", "practice"])
    assert block_roles(r).tolist() == ["calib", "test", "practice"]
    _check(r, [True, False, False], [False, True, False], [False, True, False])


def test_is_calib_per_row_with_fallback():
    r = V1.assign(block_role=["calib", "test", "practice"],
                  is_calib=pd.array([pd.NA, pd.NA, True], dtype="boolean"))
    # null is_calib -> the role decides (row 0 stays calib); True -> calib
    _check(r, [True, False, True], [False, True, False], [False, True, False])
    r["is_calib"] = pd.array([False, pd.NA, pd.NA], dtype="boolean")
    _check(r, [False, False, False], [False, True, False], [False, True, False])


def test_unknown_block_role_raises():
    with pytest.raises(ValueError, match="block_role must be one of"):
        calib_mask(V1.assign(block_role=["calibration", "test", "test"]))


def test_abstain_reason_check():
    for ok in ("censored_end", "no_calibration", "missing_feature:sparc", "no_dtw_distance"):
        assert is_abstain_reason(ok)
    for bad in ("missing_feature", "missing_feature:", "censored_end:x", "other", None):
        assert not is_abstain_reason(bad)
