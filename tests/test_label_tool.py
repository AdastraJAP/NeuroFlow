"""B9 tests: label bookkeeping, frame timing, CSV round trip, and B8 compatibility."""
import os

import numpy as np
import pandas as pd
import pytest

from neuroflow.segmentation.label_tool import (
    LABEL_COLUMNS, Labels, frame_times, load_labels, save_labels)
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
    t, src = frame_times(5, frames)
    assert src == "t_capture_ms" and np.allclose(t, frames["t_capture_ms"].iloc[:5])
    t, src = frame_times(4, fps=25.0)
    assert src == "video_fps" and np.allclose(t, [0, 40, 80, 120])
    t, _ = frame_times(len(frames) + 2, frames)          # video longer than the table -> NaN, not 0
    assert np.isnan(t[-2:]).all()
    with pytest.raises(ValueError):
        frame_times(5)


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
