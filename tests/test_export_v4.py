"""Regression checks for the G1 export boundary and saved-frame alignment."""

import csv
import json
import math

import cv2
import numpy as np
import pytest

from neuroflow.perception.export_v4 import COLUMNS, export_clip, geometry


def _landmarks():
    points = [dict(x=0.5, y=0.5, z=0.0, visibility=0.99, presence=0.99)
              for _ in range(33)]
    points[11].update(x=0.4, y=0.3)
    points[12].update(x=0.6, y=0.3)
    points[14].update(x=0.8, y=0.3)
    points[16].update(x=0.9, y=0.3)
    points[23].update(x=0.35, y=0.8)
    points[24].update(x=0.55, y=0.8)
    return points


def _fixture(tmp_path, bad_stamp=False):
    clip = "T_01"
    video = cv2.VideoWriter(str(tmp_path / f"{clip}.avi"),
                            cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
    assert video.isOpened()
    for _ in range(3):
        video.write(np.zeros((100, 100, 3), dtype=np.uint8))
    video.release()
    (tmp_path / f"{clip}.json").write_text(json.dumps(dict(
        clip=clip, arm="right", capture_frames=3, resolution=[100, 100],
        timestamp_source="monotonic capture clock", min_visibility=0.5)))
    with (tmp_path / f"{clip}.timestamps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "timestamp_ms"))
        writer.writerows(((i, t + (1 if bad_stamp and i == 1 else 0))
                          for i, t in enumerate((0, 101, 202))))
    live = []
    for i, t in enumerate((0, 101, 202)):
        valid = i != 1
        live.append(dict(frame=i, timestamp_ms=t, pose_detected=int(valid),
                         arm_elevation_deg=45 if valid else None,
                         shoulder_tilt_deg=3 if valid else None,
                         usable=int(valid), status="ok" if valid else "no_person",
                         landmarks=_landmarks() if valid else []))
    (tmp_path / f"{clip}.live_pose.jsonl").write_text(
        "\n".join(map(json.dumps, live)) + "\n")


def test_export_preserves_invalid_frame_and_exact_timestamps(tmp_path):
    _fixture(tmp_path)
    frames, provenance, manifest = export_clip(tmp_path, "T_01", "person1", "visit1", "normal", "none")
    assert tuple(frames.columns) == COLUMNS and len(COLUMNS) == 78
    assert frames.frame_idx.tolist() == [0, 1, 2]
    assert frames.t_capture_ms.tolist() == [0, 101, 202]
    assert frames.detected.tolist() == [True, False, True]
    assert math.isnan(frames.loc[1, "theta_v_deg"])
    assert math.isnan(frames.loc[1, "lm12_x_px"])
    assert frames.loc[0, "lm12_x_px"] == pytest.approx(60)
    assert provenance.source_status.tolist() == ["ok", "no_person", "ok"]
    assert manifest["frames"] == 3


def test_rejects_timestamp_disagreement(tmp_path):
    _fixture(tmp_path, bad_stamp=True)
    with pytest.raises(ValueError, match="timestamp mismatch"):
        export_clip(tmp_path, "T_01", "person1", "visit1", "normal", "none")


def test_geometry_needs_hips_and_preserves_signed_lean():
    points = _landmarks()
    measured = geometry(points, 100, 100, 0.5)
    assert measured["theta_ht_deg"] == pytest.approx(95.7105931375)
    assert measured["trunk_lean_deg"] == pytest.approx(math.degrees(math.atan2(5, 50)))
    assert measured["elbow_deg"] == pytest.approx(180)
    points[23]["y"] = 1.2
    missing = geometry(points, 100, 100, 0.5)
    assert math.isnan(missing["theta_ht_deg"])
    assert math.isnan(missing["trunk_lean_deg"])
    assert missing["elbow_deg"] == pytest.approx(180)
