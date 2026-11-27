"""Synthetic unilateral schema-v2 export and v1 compatibility regressions."""

import csv
import hashlib
import json
import math

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from neuroflow.perception.export_v4 import COLUMNS as V1_COLUMNS
from neuroflow.perception.export_v4 import export_clip as export_v1_clip
from neuroflow.perception.export_v4_v2 import (COLUMNS, SELECTION_FIELDS,
                                               export_selection)


def _landmarks(side, angle):
    points = [dict(x=0.5, y=0.5, z=0.0, visibility=0.0, presence=0.0)
              for _ in range(33)]
    for idx, x in ((11, 0.4), (12, 0.6)):
        points[idx].update(x=x, y=0.3, visibility=0.99, presence=0.99)
    shoulder, elbow, direction = (11, 13, -1) if side == "left" else (12, 14, 1)
    radians = math.radians(angle)
    points[elbow].update(x=points[shoulder]["x"] + direction * 0.2 * math.sin(radians),
                         y=0.3 + 0.2 * math.cos(radians),
                         visibility=0.99, presence=0.99)
    # Wrist and hips are intentionally unreliable; shoulder-to-elbow remains usable.
    return points


def _recording(root, clip="T_01", side="right", category="test", session="synthetic_session",
               row_arm=True):
    source = root / category
    source.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(source / f"{clip}.avi"),
                            cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
    assert video.isOpened()
    for _ in range(3):
        video.write(np.zeros((100, 100, 3), dtype=np.uint8))
    video.release()
    (source / f"{clip}.json").write_text(json.dumps(dict(
        clip=clip, category=category, arm=side, recording_schema_version=3,
        angle_definition="shoulder_to_elbow_vs_downward_image_vertical",
        capture_frames=3, resolution=[100, 100],
        timestamp_source="monotonic capture clock", min_visibility=0.5)))
    with (source / f"{clip}.timestamps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "timestamp_ms"))
        writer.writerows(((i, t) for i, t in enumerate((0, 101, 202))))
    rows = []
    for i, (t, angle) in enumerate(((0, 42.125), (101, None), (202, 91.75))):
        row = dict(frame=i, timestamp_ms=t, pose_detected=int(angle is not None),
                   arm_elevation_deg=angle, shoulder_tilt_deg=3.5 if angle is not None else None,
                   usable=int(angle is not None), status="ok" if angle is not None else "no_person",
                   landmarks=_landmarks(side, angle) if angle is not None else [])
        if row_arm:
            row["arm"] = side
        rows.append(row)
    (source / f"{clip}.live_pose.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    entry = dict(source_rel_dir=category, clip_id=clip, block_id=clip,
                 subject_id="synthetic01", session_id=session, condition="normal", level="none",
                 side=side, block_role=category, angle_col="theta_v_deg")
    return source, entry


def _selection(path, *entries):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        writer.writerows(entries)


def _export(tmp_path, entry):
    selection = tmp_path / "selection.csv"
    _selection(selection, entry)
    output = tmp_path / "output"
    manifest = export_selection(selection, tmp_path, output)
    return pd.read_parquet(output / "frames.parquet"), pd.read_parquet(
        output / "frame_provenance.parquet"), manifest, output


def test_schema_types_missing_measurements_and_consumer_metadata(tmp_path):
    _, entry = _recording(tmp_path)
    frames, provenance, manifest, output = _export(tmp_path, entry)
    assert len(COLUMNS) == 80 and COLUMNS[:78] == V1_COLUMNS
    assert tuple(frames.columns) == COLUMNS and len(frames) == 3
    assert frames.schema_version.tolist() == [2, 2, 2]
    assert frames.block_role.tolist() == ["test"] * 3
    assert frames.angle_col.tolist() == ["theta_v_deg"] * 3
    assert frames.frame_idx.tolist() == [0, 1, 2]
    assert frames.t_capture_ms.tolist() == [0, 101, 202]
    assert frames.theta_v_deg.iloc[0] == 42.125
    assert frames.theta_v_deg.iloc[2] == 91.75
    assert math.isnan(frames.theta_v_deg.iloc[1])
    assert frames.detected.tolist() == [True, False, True]
    assert frames.in_frame.tolist() == [True, False, True]
    for field in ("theta_ht_deg", "trunk_lean_deg", "elbow_deg", "frame_conf"):
        assert frames[field].isna().all(), field
    assert provenance.source_usable.tolist() == [True, False, True]
    assert provenance.source_status.tolist() == ["ok", "no_person", "ok"]
    assert provenance.shoulder_tilt_deg.iloc[0] == 3.5
    assert math.isnan(provenance.shoulder_tilt_deg.iloc[1])
    assert manifest["schema_version"] == 2
    assert manifest["clips"][0]["sources"]["avi"]["path"] == "T_01.avi"
    assert not any("normal_calib" in str(value) for value in frames.to_dict("list").values())
    # B selects the recorded angle; C receives an explicit test role, not a block-name guess.
    assert frames[frames["angle_col"].iloc[0]].notna().sum() == 2
    assert set(frames.block_role) == {"test"}
    schema = pq.read_schema(output / "frames.parquet")
    for field in ("schema_version", "frame_idx", "t_capture_ms", "img_w", "img_h"):
        assert schema.field(field).type == pa.int64()
    for field in ("subject_id", "session_id", "block_id", "side", "block_role", "angle_col"):
        assert schema.field(field).type == pa.string()
    for field in ("detected", "in_frame"):
        assert schema.field(field).type == pa.bool_()
    for field in ("theta_v_deg", "theta_ht_deg", "trunk_lean_deg", "elbow_deg", "frame_conf"):
        assert schema.field(field).type == pa.float64()


def test_practice_recording_stays_practice_with_null_string_session(tmp_path):
    _, entry = _recording(tmp_path, clip="complete_07", category="practice", session=None)
    frames, _, manifest, output = _export(tmp_path, entry)
    assert frames.block_role.tolist() == ["practice"] * 3
    assert frames.session_id.isna().all()
    assert manifest["clips"][0]["session_id"] is None
    assert pq.read_schema(output / "frames.parquet").field("session_id").type == pa.string()


def test_left_anatomical_landmarks_and_right_historical_rows(tmp_path):
    _, left = _recording(tmp_path, clip="L_01", side="left")
    frames, _, _, _ = _export(tmp_path, left)
    assert set(frames.side) == {"left"}
    assert frames.lm13_x_px.iloc[0] < frames.lm11_x_px.iloc[0]
    assert frames.theta_v_deg.iloc[0] == 42.125
    _, right = _recording(tmp_path, clip="T_02", row_arm=False)
    frames, _, _, _ = _export(tmp_path, right)
    assert set(frames.side) == {"right"}


def test_multiple_blocks_keep_distinct_roles_and_source_hashes(tmp_path):
    _, test = _recording(tmp_path, clip="T_01", category="test")
    _, practice = _recording(tmp_path, clip="complete_07", category="practice", session=None)
    _selection(tmp_path / "selection.csv", test, practice)
    manifest = export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")
    frames = pd.read_parquet(tmp_path / "output" / "frames.parquet")
    assert len(frames) == 6
    assert frames.groupby("block_id").block_role.nunique().to_dict() == {
        "T_01": 1, "complete_07": 1}
    assert frames.groupby("block_id").angle_col.nunique().to_dict() == {
        "T_01": 1, "complete_07": 1}
    assert frames.groupby("block_id").block_role.first().to_dict() == {
        "T_01": "test", "complete_07": "practice"}
    assert frames.duplicated(["block_id", "frame_idx"]).sum() == 0
    for clip in manifest["clips"]:
        for source in clip["sources"].values():
            original = tmp_path / clip["source_rel_dir"] / source["path"]
            assert hashlib.sha256(original.read_bytes()).hexdigest() == source["sha256"]


def test_v1_right_frame_fields_remain_semantically_identical(tmp_path):
    source, entry = _recording(tmp_path)
    v1, v1_provenance, _ = export_v1_clip(source, "T_01", "synthetic01", "synthetic_session", "normal", "none")
    v2, v2_provenance, _, _ = _export(tmp_path, entry)
    prefix = v2.loc[:, list(V1_COLUMNS)].copy()
    prefix["schema_version"] = 1
    pd.testing.assert_frame_equal(v1.reset_index(drop=True), prefix.reset_index(drop=True),
                                  check_dtype=False)
    pd.testing.assert_frame_equal(v1_provenance, v2_provenance, check_dtype=False)


@pytest.mark.parametrize("role", ["", "calibration", "normal_calib", "TEST"])
def test_invalid_or_missing_roles_rejected(tmp_path, role):
    _, entry = _recording(tmp_path)
    entry["block_role"] = role
    _selection(tmp_path / "selection.csv", entry)
    with pytest.raises(ValueError, match="block_role"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")


@pytest.mark.parametrize("angle", ["", "theta_ht_deg", "elbow_deg", "other"])
def test_incompatible_angle_provenance_rejected(tmp_path, angle):
    _, entry = _recording(tmp_path)
    entry["angle_col"] = angle
    _selection(tmp_path / "selection.csv", entry)
    with pytest.raises(ValueError, match="theta_v_deg"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")


def test_inconsistent_role_within_block_and_duplicate_block_rejected(tmp_path):
    _, entry = _recording(tmp_path)
    _selection(tmp_path / "selection.csv", entry, {**entry, "block_role": "practice"})
    with pytest.raises(ValueError, match="inconsistent role or angle"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")
    _selection(tmp_path / "selection.csv", entry, entry)
    with pytest.raises(ValueError, match="duplicate block_id"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")


def test_unknown_subject_or_session_placeholder_rejected(tmp_path):
    _, entry = _recording(tmp_path)
    for changed in ({"subject_id": ""}, {"subject_id": "unknown"},
                    {"session_id": "unknown"}):
        _selection(tmp_path / "selection.csv", {**entry, **changed})
        with pytest.raises(ValueError, match="verified non-null"):
            export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")


def test_selection_rejects_ambiguous_identity_and_escaping_source_path(tmp_path):
    _, entry = _recording(tmp_path)
    _selection(tmp_path / "selection.csv", {**entry, "subject_id": " synthetic01 "})
    with pytest.raises(ValueError, match="surrounding whitespace"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")
    _selection(tmp_path / "selection.csv", {**entry, "source_rel_dir": "../outside"})
    with pytest.raises(ValueError, match="escapes source root"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")
    _selection(tmp_path / "selection.csv", {**entry, "clip_id": "../T_01"})
    with pytest.raises(ValueError, match="filename stem"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")


def test_metadata_role_side_and_angle_conflicts_rejected(tmp_path):
    source, entry = _recording(tmp_path)
    for changed, error in (({"block_role": "calib"}, "block_role conflicts"),
                           ({"side": "left"}, "anatomical side conflicts")):
        _selection(tmp_path / "selection.csv", {**entry, **changed})
        with pytest.raises(ValueError, match=error):
            export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")
    meta_path = source / "T_01.json"
    meta = json.loads(meta_path.read_text())
    meta["angle_definition"] = "shoulder_to_elbow_vs_trunk"
    meta_path.write_text(json.dumps(meta))
    _selection(tmp_path / "selection.csv", entry)
    with pytest.raises(ValueError, match="angle definition"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")


def test_explicit_recording_identity_and_pose_angle_conflicts_rejected(tmp_path):
    source, entry = _recording(tmp_path)
    meta_path = source / "T_01.json"
    meta = json.loads(meta_path.read_text())
    meta["session_id"] = "another_session"
    meta_path.write_text(json.dumps(meta))
    _selection(tmp_path / "selection.csv", entry)
    with pytest.raises(ValueError, match="session_id conflicts"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")
    del meta["session_id"]
    meta_path.write_text(json.dumps(meta))
    rows_path = source / "T_01.live_pose.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["angle_definition"] = "trunk_relative"
    rows_path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="incompatible angle definition"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")


def test_left_rows_missing_arm_and_conflicting_row_arm_rejected(tmp_path):
    source, left = _recording(tmp_path, clip="L_01", side="left", row_arm=False)
    _selection(tmp_path / "selection.csv", left)
    with pytest.raises(ValueError, match="arm provenance"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")
    source, right = _recording(tmp_path, clip="T_02")
    rows_path = source / "T_02.live_pose.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["arm"] = "left"
    rows_path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    _selection(tmp_path / "selection.csv", right)
    with pytest.raises(ValueError, match="arm disagrees"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")


@pytest.mark.parametrize("failure", ["timestamp", "index", "count", "video"])
def test_alignment_errors_rejected_without_output(tmp_path, failure):
    source, entry = _recording(tmp_path)
    if failure in ("timestamp", "index"):
        path = source / "T_01.timestamps.csv"
        rows = list(csv.reader(path.open(newline="")))
        rows[2][0 if failure == "index" else 1] = "0" if failure == "index" else "102"
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
    elif failure == "count":
        path = source / "T_01.json"
        meta = json.loads(path.read_text())
        meta["capture_frames"] = 2
        path.write_text(json.dumps(meta))
    else:
        path = source / "T_01.avi"
        video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
        assert video.isOpened()
        video.write(np.zeros((100, 100, 3), dtype=np.uint8))
        video.release()
    _selection(tmp_path / "selection.csv", entry)
    with pytest.raises(ValueError, match="mismatch"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_nonmonotonic_saved_timestamp_and_source_files_unchanged(tmp_path):
    source, entry = _recording(tmp_path)
    originals = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in source.iterdir()}
    _export(tmp_path, entry)
    assert originals == {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in source.iterdir()}
    rows_path = source / "T_01.live_pose.jsonl"
    stamps_path = source / "T_01.timestamps.csv"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[2]["timestamp_ms"] = 101
    rows_path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with stamps_path.open("w", newline="") as stream:
        csv.writer(stream).writerows([("frame", "timestamp_ms"), (0, 0), (1, 101), (2, 101)])
    _selection(tmp_path / "selection.csv", entry)
    with pytest.raises(ValueError, match="timestamp mismatch"):
        export_selection(tmp_path / "selection.csv", tmp_path, tmp_path / "second_output")
