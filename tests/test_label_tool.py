"""B9 tests: label bookkeeping, frame timing, CSV round trip, and B8 compatibility."""
import os

import numpy as np
import pandas as pd
import pytest

from neuroflow.segmentation.label_tool import (
    LABEL_COLUMNS, Labels, frame_times, load_labels, save_labels, select_block)
from neuroflow.segmentation.segment import calibration_refs, segment_reps, validate

FX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_marks_cycle_start_peak_end():
    lab = Labels()
    assert [lab.mark(t) for t in (100, 200, 300, 400)] == ["start", "peak", "end", "start"]
    assert lab.next_kind() == "peak"


def test_mark_must_come_after_previous():
    lab = Labels()
    lab.mark(200)
    with pytest.raises(ValueError):
        lab.mark(150)
    with pytest.raises(ValueError):
        lab.mark(200)
    assert lab.marks == [200.0]


def test_undo():
    lab = Labels()
    assert lab.undo() is None
    lab.mark(100)
    lab.mark(200)
    assert lab.undo() == ("peak", 200.0)
    assert lab.next_kind() == "peak"


def test_unfinished_rep_is_nan_not_zero():
    lab = Labels()
    for t in (100, 200, 300, 400):
        lab.mark(t)
    df = lab.to_frame()
    assert list(df.columns) == LABEL_COLUMNS and len(df) == 2
    assert df.loc[1, "t_start_ms"] == 400 and df.loc[1, ["t_peak_ms", "t_end_ms"]].isna().all()


def test_round_trip_and_complete_only(tmp_path):
    lab = Labels()
    for t in (100, 200, 300, 400):
        lab.mark(t)
    p = tmp_path / "l.csv"
    save_labels(lab.to_frame(labeler="me"), p)
    assert len(load_labels(p)) == 1
    assert len(load_labels(p, complete_only=False)) == 2


def test_load_rejects_bad_order_and_overlap(tmp_path):
    base = {c: "" for c in LABEL_COLUMNS}
    def write(rows):
        p = tmp_path / "bad.csv"
        pd.DataFrame([{**base, "rep_idx": i, "t_start_ms": s, "t_peak_ms": k, "t_end_ms": e}
                      for i, (s, k, e) in enumerate(rows)]).to_csv(p, index=False)
        return p
    with pytest.raises(ValueError, match="start < peak < end"):
        load_labels(write([(100, 300, 200)]))
    with pytest.raises(ValueError, match="overlap"):
        load_labels(write([(100, 200, 300), (250, 400, 500)]))
    with pytest.raises(ValueError, match="missing columns"):
        p = tmp_path / "cols.csv"
        pd.DataFrame({"t_start_ms": [1]}).to_csv(p, index=False)
        load_labels(p)


def test_frame_times_from_parquet_and_fps():
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    t, src = frame_times(len(frames), frames)
    assert src == "t_capture_ms" and np.allclose(t, frames.sort_values("frame_idx")["t_capture_ms"])
    t, src = frame_times(4, fps=25.0)
    assert src == "video_fps" and np.allclose(t, [0, 40, 80, 120])
    with pytest.raises(ValueError):
        frame_times(5)


def test_frame_times_needs_a_complete_mapping():
    """Adastra's PR #3 review: every decoded video frame needs exactly one finite time."""
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    n = len(frames)
    with pytest.raises(ValueError, match="exactly 0"):
        frame_times(n + 2, frames)                         # video longer than its frame table
    with pytest.raises(ValueError, match="exactly 0"):
        frame_times(n - 2, frames)                         # table longer than the video
    with pytest.raises(ValueError, match="exactly 0"):
        frame_times(n, frames.assign(frame_idx=frames["frame_idx"] + 1))   # not zero-based
    with pytest.raises(ValueError, match="duplicate"):
        frame_times(n, frames.assign(frame_idx=np.r_[0, frames["frame_idx"].to_numpy()[:-1]]))
    gap = frames.copy()
    gap.loc[gap.index[10], "t_capture_ms"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        frame_times(n, gap)
    back = frames.copy()
    back.loc[back["frame_idx"] == 10, "t_capture_ms"] = 0.0
    with pytest.raises(ValueError, match="strictly increasing"):
        frame_times(n, back)


def test_mark_rejects_non_finite_times():
    lab = Labels()
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="not a finite"):
            lab.mark(bad)
    assert lab.marks == []


def test_labels_feed_b8_comparison(tmp_path):
    """Labels made at the truth times load straight into segment.validate (what B8 will do)."""
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    truth = pd.read_csv(os.path.join(FX, "mock_frames_normal_truth.csv"))
    lab = Labels()
    for r in truth.itertuples():
        for t in (r.t_start_ms, r.t_peak_ms, r.t_end_ms):
            lab.mark(t)
    p = tmp_path / "labels.csv"
    save_labels(lab.to_frame(subject_id="mock01", labeler="test"), p)
    rest, rom = calibration_refs()["mock01"]
    ok, n, _ = validate(frames, load_labels(p), reps=segment_reps(frames, rom, rest))
    assert ok and n == len(truth)


def test_select_block():
    a = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    b = pd.read_parquet(os.path.join(FX, "mock_frames_slow.parquet"))
    both = pd.concat([a, b], ignore_index=True)
    assert len(select_block(both, "slow")) == len(b)
    assert len(select_block(a)) == len(a)                       # single clip: no --block needed
    with pytest.raises(ValueError, match="pass --block"):
        select_block(both)
    with pytest.raises(ValueError, match="available"):
        select_block(both, "T_99")
    t, _ = frame_times(len(b), select_block(both, "slow"))      # frame_idx restarts per clip
    assert np.allclose(t, b.sort_values("frame_idx")["t_capture_ms"])


def test_load_rejects_infinite_non_numeric_and_unordered_partial(tmp_path):
    """Adastra's PR #3 re-review: imported boundaries must be finite and ordered."""
    base = {c: "" for c in LABEL_COLUMNS}
    def write(rows):
        p = tmp_path / "bad.csv"
        pd.DataFrame([{**base, "rep_idx": i, "t_start_ms": s, "t_peak_ms": k, "t_end_ms": e}
                      for i, (s, k, e) in enumerate(rows)]).to_csv(p, index=False)
        return p
    for row in [(100, 200, np.inf), (-np.inf, 200, 300), (100, np.inf, 300)]:
        with pytest.raises(ValueError, match="infinite"):
            load_labels(write([row]))
    with pytest.raises(ValueError, match="not numbers"):
        load_labels(write([(100, "abc", 300)]))
    # unfinished rep (no end yet): its marked points must still be in order
    with pytest.raises(ValueError, match="start < peak < end"):
        load_labels(write([(300, 200, np.nan)]), complete_only=False)
    with pytest.raises(ValueError, match="start < peak < end"):
        load_labels(write([(100, 100, 300)]))               # equal is not "after"
    ok = load_labels(write([(100, 200, 300), (400, 500, np.nan)]), complete_only=False)
    assert len(ok) == 2


# ---- points outside the clip (G1: the clip ends before the arm is back at rest) ----

def test_end_outside_clip_round_trip(tmp_path):
    lab = Labels()
    for t in (100, 200, 300, 400, 500):
        lab.mark(t)
    assert lab.mark_outside() == "end"
    with pytest.raises(ValueError, match="no rep can follow"):
        lab.mark(600)
    df = lab.to_frame(block_id="T_05")
    assert list(df["outside_clip"]) == ["", "end"] and np.isnan(df["t_end_ms"].iloc[1])
    p = tmp_path / "l.csv"
    save_labels(df, p)
    back = load_labels(p)                                    # complete_only keeps the flagged rep
    assert len(back) == 2 and back["outside_clip"].iloc[1] == "end"


def test_start_outside_clip_only_for_first_rep():
    lab = Labels()
    assert lab.mark_outside() == "start"
    lab.mark(200)                                            # peak
    lab.mark(300)                                            # end
    with pytest.raises(ValueError, match="only the first rep's start"):
        lab.mark_outside()
    assert list(lab.to_frame()["outside_clip"]) == ["start"]
    other = Labels()
    other.mark(100)
    with pytest.raises(ValueError, match="peak can't be outside"):
        other.mark_outside()


def test_unfinished_rep_is_not_outside_clip(tmp_path):
    lab = Labels()
    for t in (100, 200):
        lab.mark(t)                                          # end simply not marked yet
    df = lab.to_frame()
    assert df["outside_clip"].iloc[0] == ""
    p = tmp_path / "l.csv"
    save_labels(df, p)
    assert len(load_labels(p)) == 0                          # dropped: not complete
    assert len(load_labels(p, complete_only=False)) == 1


def test_load_rejects_bad_outside_flags(tmp_path):
    def write(rows):
        p = tmp_path / "bad.csv"
        pd.DataFrame([{c: "" for c in LABEL_COLUMNS} | dict(zip(
            ["rep_idx", "t_start_ms", "t_peak_ms", "t_end_ms", "outside_clip"], (i, *r)))
            for i, r in enumerate(rows)]).to_csv(p, index=False)
        return p
    with pytest.raises(ValueError, match="has a time"):
        load_labels(write([(100, 200, 300, "end")]))
    with pytest.raises(ValueError, match="last rep's end"):
        load_labels(write([(100, 200, np.nan, "end"), (400, 500, 600, "")]))
    with pytest.raises(ValueError, match="must be"):
        load_labels(write([(100, 200, 300, "peak")]))


def test_old_files_without_the_column_still_load(tmp_path):
    p = tmp_path / "old.csv"
    pd.DataFrame([{"rep_idx": 0, "t_start_ms": 100, "t_peak_ms": 200, "t_end_ms": 300,
                   "subject_id": "P001", "session_id": "S001", "block_id": "T_01",
                   "labeler": "x", "time_source": "t_capture_ms"}]).to_csv(p, index=False)
    assert load_labels(p)["outside_clip"].iloc[0] == ""
