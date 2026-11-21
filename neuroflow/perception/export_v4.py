"""Export saved V4 live measurements into the existing v1 frames contract.

Run ``python -m neuroflow.perception.export_v4 --help``. No pose inference occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


LANDMARK_IDS = (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24)
LANDMARK_FIELDS = ("x_px", "y_px", "z", "vis", "pres")
BASE = ("schema_version", "subject_id", "session_id", "block_id", "condition",
        "level", "side", "frame_idx", "t_capture_ms", "img_w", "img_h",
        "pose_model", "pose_model_ver", "detected")
DERIVED = ("theta_ht_deg", "theta_v_deg", "trunk_lean_deg", "elbow_deg",
           "hike_ratio", "in_frame", "jitter", "arm_len_ratio", "frame_conf")
COLUMNS = (*BASE, *(f"lm{i}_{field}" for i in LANDMARK_IDS for field in LANDMARK_FIELDS), *DERIVED)
FLOAT_COLUMNS = (*(f"lm{i}_{field}" for i in LANDMARK_IDS for field in LANDMARK_FIELDS),
                 *(x for x in DERIVED if x != "in_frame"))
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "NeuroFlow_Shoulder_to_Elbow_and_ShoulderLine_tilt_approach" / "clips" / "seated" / "segmented_v1" / "test"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid(lm: dict, min_visibility: float) -> bool:
    if any(lm.get(key) is None for key in ("x", "y", "visibility", "presence")):
        return False
    return (all(math.isfinite(float(lm[key])) for key in ("x", "y", "visibility", "presence"))
            and 0 <= lm["x"] <= 1 and 0 <= lm["y"] <= 1
            and lm["visibility"] >= min_visibility and lm["presence"] >= min_visibility)


def _point(lm: dict, width: int, height: int) -> np.ndarray:
    return np.array([lm["x"] * width, lm["y"] * height], dtype=float)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return math.nan
    cosine = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def geometry(landmarks: list[dict], width: int, height: int, min_visibility: float) -> dict:
    """Compute right arm against downward trunk, signed lean, and elbow angle.

    Angles use 2D image pixels; positive lean means towards anatomical right.
    Missing/out-of-frame/low-confidence inputs produce NaN.
    """
    result = {"theta_ht_deg": math.nan, "trunk_lean_deg": math.nan,
              "elbow_deg": math.nan}
    if len(landmarks) != 33:
        return result
    ok = {i: _valid(landmarks[i], min_visibility) for i in (11, 12, 14, 16, 23, 24)}
    pt = {i: _point(landmarks[i], width, height) for i in ok if ok[i]}
    if all(ok[i] for i in (12, 14, 16)):
        result["elbow_deg"] = _angle(pt[12] - pt[14], pt[16] - pt[14])
    if all(ok[i] for i in (11, 12, 14, 23, 24)):
        shoulder_mid = (pt[11] + pt[12]) / 2
        hip_mid = (pt[23] + pt[24]) / 2
        trunk_down = hip_mid - shoulder_mid
        result["theta_ht_deg"] = _angle(pt[14] - pt[12], trunk_down)
        body_right = np.sign(pt[12][0] - pt[11][0])
        if body_right != 0:
            result["trunk_lean_deg"] = float(np.degrees(np.arctan2(
                body_right * (shoulder_mid[0] - hip_mid[0]),
                hip_mid[1] - shoulder_mid[1])))
    return result


def _video_info(path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    count = 0
    width = height = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if count == 0:
                height, width = image.shape[:2]
            elif image.shape[:2] != (height, width):
                raise ValueError(f"video resolution changes: {path}")
            count += 1
    finally:
        capture.release()
    return count, width, height


def export_clip(source: Path, clip_id: str, subject_id: str, session_id: str,
                condition: str, level: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = {suffix: source / f"{clip_id}.{suffix}" for suffix in
             ("avi", "json", "timestamps.csv", "live_pose.jsonl")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads(paths["json"].read_text(encoding="utf-8"))
    if meta.get("clip") != clip_id or meta.get("arm") != "right":
        raise ValueError(f"wrong clip or arm in metadata: {clip_id}")
    with paths["timestamps.csv"].open(newline="", encoding="utf-8") as stream:
        times = list(csv.DictReader(stream))
    with paths["live_pose.jsonl"].open(encoding="utf-8") as stream:
        live = [json.loads(line) for line in stream if line.strip()]
    video_count, width, height = _video_info(paths["avi"])
    if not (video_count == len(times) == len(live) == meta["capture_frames"]):
        raise ValueError(f"frame count mismatch for {clip_id}: video={video_count}, "
                         f"times={len(times)}, pose={len(live)}, meta={meta['capture_frames']}")
    if [width, height] != meta["resolution"]:
        raise ValueError(f"video resolution mismatch for {clip_id}")
    if meta.get("timestamp_source") != "monotonic capture clock":
        raise ValueError(f"unverified timestamp source for {clip_id}")
    min_visibility = float(meta["min_visibility"])
    frames, provenance = [], []
    previous_time = -1
    for idx, (stamp, saved) in enumerate(zip(times, live)):
        t = int(saved["timestamp_ms"])
        if (int(saved["frame"]) != idx or int(stamp["frame"]) != idx
                or int(stamp["timestamp_ms"]) != t or t <= previous_time):
            raise ValueError(f"frame index or timestamp mismatch in {clip_id} row {idx}")
        previous_time = t
        landmarks = saved["landmarks"]
        if len(landmarks) not in (0, 33):
            raise ValueError(f"unexpected landmark count in {clip_id} row {idx}")
        detected = bool(saved["pose_detected"])
        if detected != bool(landmarks):
            raise ValueError(f"pose detection disagrees with landmarks in {clip_id} row {idx}")
        row = dict(schema_version=1, subject_id=subject_id, session_id=session_id,
                   block_id=clip_id, condition=condition, level=level, side="right",
                   frame_idx=idx, t_capture_ms=t, img_w=width, img_h=height,
                   pose_model="mediapipe_pose_landmarker_lite", pose_model_ver="unrecorded",
                   detected=detected)
        for lm_id in LANDMARK_IDS:
            lm = landmarks[lm_id] if detected else None
            for field, source_field in zip(LANDMARK_FIELDS,
                                           ("x", "y", "z", "visibility", "presence")):
                value = lm.get(source_field, math.nan) if lm else math.nan
                if value is None:
                    value = math.nan
                if field == "x_px" and math.isfinite(value):
                    value *= width
                if field == "y_px" and math.isfinite(value):
                    value *= height
                row[f"lm{lm_id}_{field}"] = value
        row.update(geometry(landmarks, width, height, min_visibility))
        row["theta_v_deg"] = (float(saved["arm_elevation_deg"])
                              if saved["usable"] and saved["arm_elevation_deg"] is not None
                              else math.nan)
        row["in_frame"] = (bool(_valid(landmarks[12], min_visibility)
                                and _valid(landmarks[14], min_visibility)) if detected else False)
        for field in ("hike_ratio", "jitter", "arm_len_ratio", "frame_conf"):
            row[field] = math.nan
        frames.append(row)
        provenance.append(dict(block_id=clip_id, frame_idx=idx, t_capture_ms=t,
                               source_usable=bool(saved["usable"]),
                               source_status=saved["status"],
                               shoulder_tilt_deg=(saved["shoulder_tilt_deg"]
                                                  if saved["shoulder_tilt_deg"] is not None
                                                  else math.nan)))
    frame_table = pd.DataFrame(frames, columns=COLUMNS)
    frame_table[list(FLOAT_COLUMNS)] = frame_table[list(FLOAT_COLUMNS)].astype("float64")
    manifest = dict(clip_id=clip_id, subject_id=subject_id, session_id=session_id,
                    condition=condition, level=level, frames=video_count,
                    timestamp_first_ms=int(live[0]["timestamp_ms"]),
                    timestamp_last_ms=int(live[-1]["timestamp_ms"]),
                    valid_theta_v=int(frame_table.theta_v_deg.notna().sum()),
                    valid_theta_ht=int(frame_table.theta_ht_deg.notna().sum()),
                    valid_trunk_lean=int(frame_table.trunk_lean_deg.notna().sum()),
                    sources={key: {"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT)
                                                else path.resolve()), "sha256": _sha256(path)}
                             for key, path in paths.items()})
    return frame_table, pd.DataFrame(provenance), manifest


def export_selection(selection: Path, source: Path, output: Path) -> dict:
    with selection.open(newline="", encoding="utf-8") as stream:
        entries = list(csv.DictReader(stream))
    required = ("clip_id", "subject_id", "session_id", "condition", "level")
    if len(entries) != 3 or {entry.get("clip_id") for entry in entries} != {"T_01", "T_02", "T_05"}:
        raise ValueError("selection must contain exactly T_01, T_02 and T_05")
    if any(not all(entry.get(field) for field in required) for entry in entries):
        raise ValueError(f"selection requires {required}; no guessed identifiers")
    pieces = [export_clip(source, *(entry[field] for field in required)) for entry in entries]
    frames = pd.concat([part[0] for part in pieces], ignore_index=True)
    provenance = pd.concat([part[1] for part in pieces], ignore_index=True)
    if frames.duplicated(["block_id", "frame_idx"]).any():
        raise ValueError("duplicate clip/frame key")
    output.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(output / "frames.parquet", index=False)
    provenance.to_parquet(output / "frame_provenance.parquet", index=False)
    manifest = {"schema_version": 1, "frames_parquet": "frames.parquet",
                "provenance_parquet": "frame_provenance.parquet",
                "clips": [part[2] for part in pieces]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True,
                        help="CSV with clip_id,subject_id,session_id,condition,level")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "g1")
    args = parser.parse_args()
    manifest = export_selection(args.selection, args.source, args.output)
    print(json.dumps({"output": str(args.output), "clips": manifest["clips"]}, indent=2))


if __name__ == "__main__":
    main()
