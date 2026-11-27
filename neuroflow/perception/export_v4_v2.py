"""Export saved unilateral V4 captures to the 80-column frames schema v2.

Run ``python -m neuroflow.perception.export_v4_v2 --help``. This module never
reruns MediaPipe. The G1 schema-v1 exporter remains a separate entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .export_v4 import (COLUMNS as V1_COLUMNS, FLOAT_COLUMNS, LANDMARK_FIELDS,
                        LANDMARK_IDS, ROOT, _angle, _point, _sha256, _valid,
                        _video_info, geometry as right_geometry)


SCHEMA_VERSION = 2
COLUMNS = (*V1_COLUMNS, "block_role", "angle_col")
BLOCK_ROLES = frozenset(("calib", "practice", "test"))
ANGLE_COL = "theta_v_deg"
V4_ANGLE_DEFINITION = "shoulder_to_elbow_vs_downward_image_vertical"
SELECTION_FIELDS = ("source_rel_dir", "clip_id", "block_id", "subject_id",
                    "session_id", "condition", "level", "side", "block_role",
                    "angle_col")
STRING_COLUMNS = ("subject_id", "session_id", "block_id", "condition", "level",
                  "side", "pose_model", "pose_model_ver", "block_role", "angle_col")
INT_COLUMNS = ("schema_version", "frame_idx", "t_capture_ms", "img_w", "img_h")
DEFAULT_SOURCE_ROOT = (ROOT / "NeuroFlow_Shoulder_to_Elbow_and_ShoulderLine_tilt_approach"
                       / "clips" / "seated")


def _identifier(value: str | None, field: str) -> str:
    if value is None or not value.strip() or value.strip().lower() in ("unknown", "null", "none"):
        raise ValueError(f"{field} requires a verified non-null value")
    if value != value.strip():
        raise ValueError(f"{field} has surrounding whitespace; correct the selection")
    if field == "clip_id" and Path(value).name != value:
        raise ValueError("clip_id must be a filename stem, not a path")
    return value


def _selection_rows(selection: Path, source_root: Path) -> list[dict]:
    with selection.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
                or set(reader.fieldnames) != set(SELECTION_FIELDS)):
            raise ValueError(f"selection columns must be exactly {SELECTION_FIELDS}")
        entries = list(reader)
    if not entries:
        raise ValueError("selection must contain at least one recording")
    seen = {}
    for entry in entries:
        for field in ("clip_id", "block_id", "subject_id"):
            entry[field] = _identifier(entry[field], field)
        for field in ("condition", "level"):
            if entry[field] is None or not entry[field].strip():
                raise ValueError(f"{field} requires a non-null value")
            entry[field] = entry[field].strip()
        raw_session = entry["session_id"]
        entry["session_id"] = (_identifier(raw_session, "session_id")
                               if raw_session and raw_session.strip() else None)
        if entry["side"] not in ("left", "right"):
            raise ValueError(f"{entry['block_id']}: side must be anatomical left or right")
        if entry["block_role"] not in BLOCK_ROLES:
            raise ValueError(f"{entry['block_id']}: block_role must be calib, practice or test")
        if entry["angle_col"] != ANGLE_COL or entry["angle_col"] not in COLUMNS:
            raise ValueError(f"{entry['block_id']}: V4 supplies theta_v_deg, not {entry['angle_col']!r}")
        if entry["block_id"] in seen:
            prior = seen[entry["block_id"]]
            if any(prior[field] != entry[field] for field in ("block_role", "angle_col")):
                raise ValueError(f"{entry['block_id']}: inconsistent role or angle within block")
            raise ValueError(f"duplicate block_id: {entry['block_id']}")
        seen[entry["block_id"]] = entry
        relative = Path(entry["source_rel_dir"] or "")
        if relative.is_absolute() or not entry["source_rel_dir"]:
            raise ValueError(f"{entry['block_id']}: source_rel_dir must be relative")
        source = (source_root / relative).resolve()
        if not source.is_relative_to(source_root.resolve()):
            raise ValueError(f"{entry['block_id']}: source_rel_dir escapes source root")
        entry["source_rel_dir"] = relative.as_posix()
        entry["source"] = source
    return entries


def _left_geometry(landmarks: list[dict], width: int, height: int,
                   min_visibility: float) -> dict:
    """Apply the v1 projected trunk/elbow formulas to anatomical left joints."""
    result = {"theta_ht_deg": math.nan, "trunk_lean_deg": math.nan,
              "elbow_deg": math.nan}
    if len(landmarks) != 33:
        return result
    ok = {i: _valid(landmarks[i], min_visibility) for i in (11, 12, 13, 15, 23, 24)}
    pt = {i: _point(landmarks[i], width, height) for i in ok if ok[i]}
    if all(ok[i] for i in (11, 13, 15)):
        result["elbow_deg"] = _angle(pt[11] - pt[13], pt[15] - pt[13])
    if all(ok[i] for i in (11, 12, 13, 23, 24)):
        shoulder_mid = (pt[11] + pt[12]) / 2
        hip_mid = (pt[23] + pt[24]) / 2
        result["theta_ht_deg"] = _angle(pt[13] - pt[11], hip_mid - shoulder_mid)
        body_right = np.sign(pt[12][0] - pt[11][0])
        if body_right != 0:
            result["trunk_lean_deg"] = float(np.degrees(np.arctan2(
                body_right * (shoulder_mid[0] - hip_mid[0]),
                hip_mid[1] - shoulder_mid[1])))
    return result


def _integer(value, field: str) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError(f"{field} must be an integer")


def _recorded_arm(live: list[dict], meta: dict, side: str, clip_id: str) -> None:
    """Permit armless pre-A5 schema-3 rows only with explicit right metadata."""
    arms = [saved.get("arm") for saved in live]
    if any(arm not in (None, side) for arm in arms):
        raise ValueError(f"{clip_id}: live-pose arm disagrees with recording metadata")
    if any(arm is None for arm in arms):
        if (side != "right" or meta.get("recording_schema_version") != 3
                or any(arm is not None for arm in arms)):
            raise ValueError(f"{clip_id}: ambiguous or incomplete live-pose arm provenance")


def _typed_frames(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=COLUMNS)
    for field in STRING_COLUMNS:
        table[field] = table[field].astype("string")
    for field in INT_COLUMNS:
        table[field] = table[field].astype("int64")
    table[list(FLOAT_COLUMNS)] = table[list(FLOAT_COLUMNS)].astype("float64")
    table[["detected", "in_frame"]] = table[["detected", "in_frame"]].astype(bool)
    return table


def _write_parquet(table: pd.DataFrame, path: Path, string_columns: tuple[str, ...]) -> None:
    """Pin Arrow's string type, including when a whole session column is null."""
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    schema = pa.schema([pa.field(field.name, pa.string(), nullable=field.nullable)
                        if field.name in string_columns else field
                        for field in arrow.schema], metadata=arrow.schema.metadata)
    pq.write_table(arrow.cast(schema), path)


def export_clip(entry: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate one original recording and export every captured frame."""
    clip_id, block_id, side = entry["clip_id"], entry["block_id"], entry["side"]
    paths = {suffix: entry["source"] / f"{clip_id}.{suffix}" for suffix in
             ("avi", "json", "timestamps.csv", "live_pose.jsonl")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads(paths["json"].read_text(encoding="utf-8"))
    if meta.get("clip") != clip_id or meta.get("arm") != side:
        raise ValueError(f"{clip_id}: clip or anatomical side conflicts with metadata")
    for field in ("subject_id", "session_id", "condition", "level"):
        if field in meta and meta[field] != entry[field]:
            raise ValueError(f"{clip_id}: {field} conflicts with recording metadata")
    if meta.get("category") != entry["block_role"]:
        raise ValueError(f"{clip_id}: block_role conflicts with original recording category")
    if meta.get("angle_definition") != V4_ANGLE_DEFINITION:
        raise ValueError(f"{clip_id}: unsupported or missing V4 angle definition")
    if meta.get("timestamp_source") != "monotonic capture clock":
        raise ValueError(f"{clip_id}: unverified timestamp source")
    min_visibility = float(meta["min_visibility"])
    if not math.isfinite(min_visibility) or not 0 <= min_visibility <= 1:
        raise ValueError(f"{clip_id}: invalid min_visibility")
    with paths["timestamps.csv"].open(newline="", encoding="utf-8") as stream:
        stamps = list(csv.DictReader(stream))
    with paths["live_pose.jsonl"].open(encoding="utf-8") as stream:
        live = [json.loads(line) for line in stream if line.strip()]
    count, width, height = _video_info(paths["avi"])
    if (count == 0 or count != len(stamps) or count != len(live)
            or count != _integer(meta.get("capture_frames"), "capture_frames")):
        raise ValueError(f"{clip_id}: video, timestamp, pose or metadata frame count mismatch")
    if meta.get("resolution") != [width, height]:
        raise ValueError(f"{clip_id}: decoded resolution conflicts with metadata")
    _recorded_arm(live, meta, side, clip_id)
    frames, provenance = [], []
    previous_time = -1
    shoulder, elbow = (11, 13) if side == "left" else (12, 14)
    for idx, (stamp, saved) in enumerate(zip(stamps, live)):
        if saved.get("angle_definition", V4_ANGLE_DEFINITION) != V4_ANGLE_DEFINITION:
            raise ValueError(f"{clip_id}: pose row has incompatible angle definition at row {idx}")
        t = _integer(saved.get("timestamp_ms"), "saved timestamp_ms")
        if (_integer(saved.get("frame"), "saved frame") != idx
                or _integer(stamp.get("frame"), "timestamp frame") != idx
                or _integer(stamp.get("timestamp_ms"), "timestamp_ms") != t
                or t <= previous_time):
            raise ValueError(f"{clip_id}: frame index or timestamp mismatch at row {idx}")
        previous_time = t
        landmarks = saved.get("landmarks")
        if not isinstance(landmarks, list) or len(landmarks) not in (0, 33):
            raise ValueError(f"{clip_id}: invalid landmark count at row {idx}")
        if saved.get("pose_detected") not in (0, 1) or type(saved.get("pose_detected")) is not int:
            raise ValueError(f"{clip_id}: invalid pose detection at row {idx}")
        detected = bool(saved["pose_detected"])
        if detected != bool(landmarks):
            raise ValueError(f"{clip_id}: pose detection disagrees with landmarks at row {idx}")
        if saved.get("usable") not in (0, 1) or type(saved.get("usable")) is not int:
            raise ValueError(f"{clip_id}: invalid tracking decision at row {idx}")
        angle = saved.get("arm_elevation_deg")
        if (bool(saved["usable"]) != (angle is not None)
                or (angle is not None and (type(angle) not in (int, float)
                                           or not math.isfinite(angle)))):
            raise ValueError(f"{clip_id}: inconsistent saved elevation at row {idx}")
        tilt = saved.get("shoulder_tilt_deg")
        if tilt is not None and (type(tilt) not in (int, float) or not math.isfinite(tilt)):
            raise ValueError(f"{clip_id}: invalid saved shoulder tilt at row {idx}")
        status = saved.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"{clip_id}: missing tracking status at row {idx}")
        row = dict(schema_version=SCHEMA_VERSION, subject_id=entry["subject_id"],
                   session_id=entry["session_id"], block_id=block_id,
                   condition=entry["condition"], level=entry["level"], side=side,
                   frame_idx=idx, t_capture_ms=t, img_w=width, img_h=height,
                   pose_model="mediapipe_pose_landmarker_lite", pose_model_ver="unrecorded",
                   detected=detected, block_role=entry["block_role"], angle_col=ANGLE_COL)
        for lm_id in LANDMARK_IDS:
            lm = landmarks[lm_id] if detected else None
            if lm is not None and not isinstance(lm, dict):
                raise ValueError(f"{clip_id}: invalid landmark {lm_id} at row {idx}")
            for field, source_field in zip(LANDMARK_FIELDS,
                                           ("x", "y", "z", "visibility", "presence")):
                value = lm.get(source_field, math.nan) if lm else math.nan
                if value is None:
                    value = math.nan
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{clip_id}: nonnumeric landmark {lm_id} at row {idx}")
                if field == "x_px" and math.isfinite(value):
                    value *= width
                if field == "y_px" and math.isfinite(value):
                    value *= height
                row[f"lm{lm_id}_{field}"] = value
        row.update((_left_geometry if side == "left" else right_geometry)(
            landmarks, width, height, min_visibility))
        row[ANGLE_COL] = float(angle) if angle is not None else math.nan
        row["in_frame"] = (bool(_valid(landmarks[shoulder], min_visibility)
                                and _valid(landmarks[elbow], min_visibility)) if detected else False)
        for field in ("hike_ratio", "jitter", "arm_len_ratio", "frame_conf"):
            row[field] = math.nan
        frames.append(row)
        provenance.append(dict(block_id=block_id, frame_idx=idx, t_capture_ms=t,
                               source_usable=bool(saved["usable"]), source_status=status,
                               shoulder_tilt_deg=tilt if tilt is not None else math.nan))
    frame_table = _typed_frames(frames)
    source_table = pd.DataFrame(provenance)
    source_table["block_id"] = source_table["block_id"].astype("string")
    source_table["source_status"] = source_table["source_status"].astype("string")
    source_table[["frame_idx", "t_capture_ms"]] = source_table[["frame_idx", "t_capture_ms"]].astype("int64")
    source_table["source_usable"] = source_table["source_usable"].astype(bool)
    source_table["shoulder_tilt_deg"] = source_table["shoulder_tilt_deg"].astype("float64")
    manifest = dict(clip_id=clip_id, block_id=block_id, subject_id=entry["subject_id"],
                    session_id=entry["session_id"], side=side, block_role=entry["block_role"],
                    angle_col=ANGLE_COL, condition=entry["condition"], level=entry["level"],
                    source_rel_dir=entry["source_rel_dir"], recording_category=meta["category"],
                    frames=count, timestamp_first_ms=int(frame_table.t_capture_ms.iloc[0]),
                    timestamp_last_ms=int(frame_table.t_capture_ms.iloc[-1]),
                    valid_theta_v=int(frame_table.theta_v_deg.notna().sum()),
                    sources={suffix: {"path": f"{clip_id}.{suffix}", "sha256": _sha256(path)}
                             for suffix, path in paths.items()})
    return frame_table, source_table, manifest


def export_selection(selection: Path, source_root: Path, output: Path) -> dict:
    entries = _selection_rows(selection, source_root)
    pieces = [export_clip(entry) for entry in entries]
    frames = pd.concat([piece[0] for piece in pieces], ignore_index=True)
    provenance = pd.concat([piece[1] for piece in pieces], ignore_index=True)
    if (frames.duplicated(["block_id", "frame_idx"]).any()
            or provenance.duplicated(["block_id", "frame_idx"]).any()):
        raise ValueError("duplicate block/frame identity")
    output.mkdir(parents=True, exist_ok=True)
    _write_parquet(frames, output / "frames.parquet", STRING_COLUMNS)
    _write_parquet(provenance, output / "frame_provenance.parquet",
                   ("block_id", "source_status"))
    manifest = {"schema_version": SCHEMA_VERSION, "frames_parquet": "frames.parquet",
                "provenance_parquet": "frame_provenance.parquet",
                "clips": [piece[2] for piece in pieces]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True,
                        help="verified CSV with source_rel_dir, clip_id, block_id, identity and role")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "frames_v2")
    args = parser.parse_args()
    manifest = export_selection(args.selection, args.source_root, args.output)
    print(json.dumps({"output": str(args.output), "clips": manifest["clips"]}, indent=2))


if __name__ == "__main__":
    main()
