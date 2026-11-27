"""Offline processing shared by check_clip and step5_score.

One row per ORIGINAL frame. Missing angles are empty/NaN in CSV, never zero.
"""
import csv
import json
import math
import statistics
from pathlib import Path

import cv2
import nf_core as nf

JOINT_NAMES = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_hip", "right_hip")
LIVE_POSE_SCHEMA = 3


def measure_frame(index, timestamp, landmarks, settings, width, height):
    visibility = settings['min_visibility']
    angle, usable = nf.shoulder_angle(landmarks, settings['arm'], width, height, visibility)
    return {
        'frame': index, 'timestamp_ms': timestamp, 'arm': settings['arm'],
        'pose_detected': int(landmarks is not None),
        'arm_elevation_deg': angle if usable else None,
        'shoulder_tilt_deg': nf.shoulder_tilt(landmarks, width, height, visibility),
        'usable': int(usable),
        'status': 'ok' if usable else ('no_person' if landmarks is None else 'unreliable_arm_joints'),
    }


def capture_landmarks(landmarks):
    if landmarks is None:
        return None
    result = []
    for landmark in landmarks:
        point = {key: getattr(landmark, key, None) for key in ('x', 'y', 'z', 'visibility', 'presence')}
        result.append({key: value if value is not None and math.isfinite(value) else None
                       for key, value in point.items()})
    return result


def live_pose_path(path):
    return Path(path).with_suffix('.live_pose.jsonl')


def save_live_pose(path, rows):
    with live_pose_path(path).open('x', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, allow_nan=False) + '\n')


def load_live_pose(path, settings, metadata):
    if metadata.get('recording_schema_version') != LIVE_POSE_SCHEMA:
        raise ValueError(f'{path.name}: exact live pose was not saved by this recorder')
    if metadata.get('min_visibility') != settings['min_visibility']:
        raise ValueError(f'{path.name}: recorded visibility setting differs from the current setup')
    with live_pose_path(path).open(encoding='utf-8') as source:
        rows = [json.loads(line) for line in source]
    # Pre-A5 schema-3 rows omitted arm, but their recording metadata saved
    # anatomical right. Schema alone cannot identify an armless recording.
    metadata_arm = metadata.get('arm')
    row_arms = [row.get('arm') for row in rows]
    if metadata_arm not in (None, 'left', 'right') or any(
            arm not in (None, 'left', 'right') for arm in row_arms):
        raise ValueError(f'{path.name}: invalid recorded arm provenance')
    if any(arm is None for arm in row_arms):
        if metadata_arm != 'right' or any(arm is not None for arm in row_arms):
            raise ValueError(f'{path.name}: incomplete or ambiguous recorded arm provenance')
        recorded_arm = 'right'
    else:
        explicit_arms = set(row_arms)
        if len(explicit_arms) != 1:
            raise ValueError(f'{path.name}: conflicting recorded arm provenance')
        recorded_arm = explicit_arms.pop()
        if metadata_arm is not None and metadata_arm != recorded_arm:
            raise ValueError(f'{path.name}: conflicting recorded arm provenance')
    if recorded_arm != settings['arm']:
        raise ValueError(f'{path.name}: recorded arm differs from the current setup')
    times = nf.clip_timestamps(path, None)
    if (not rows or len(rows) != metadata.get('capture_frames')
            or [row['timestamp_ms'] for row in rows] != times
            or rows[-1]['timestamp_ms'] != metadata.get('termination_elapsed_ms')):
        raise ValueError(f'{path.name}: live pose does not match the capture timestamps and frame count')
    for index, row in enumerate(rows):
        angle = row['arm_elevation_deg']
        if (row['frame'] != index or type(row['timestamp_ms']) is not int or row['usable'] not in (0, 1)
                or bool(row['usable']) != (angle is not None)
                or (angle is not None and not math.isfinite(angle))
                or 'landmarks' not in row):
            raise ValueError(f'{path.name}: invalid live pose row {index}')
    return summarize_frames(path, rows, settings, 'recorded_monotonic', 'live_capture')


def process_clip(path, settings, snapshots=False, require_measured_time=False):
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps < 1:
        fps = 30.0
    try:
        times = nf.clip_timestamps(path, fps)
        if require_measured_time and (not times or times[0] < 0 or
                                      any(right <= left for left, right in zip(times, times[1:]))):
            raise ValueError(f'Missing or invalid measured frame times: {path}')
    except (ValueError, KeyError, OverflowError):
        cap.release()
        raise ValueError(f'Missing or invalid measured frame times: {path}') from None
    rows, shots = [], {}
    first, lowest, highest, last = None, None, None, None
    previous_ts = -1
    try:
        with nf.make_landmarker() as detector:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = nf.frame_timestamp(idx, times, fps, previous_ts)
                previous_ts = ts
                lms = nf.detect(detector, frame, ts)
                row = measure_frame(idx, ts, lms, settings, w, h)
                angle, usable = row['arm_elevation_deg'], row['usable']
                for name in JOINT_NAMES:
                    idx_joint = nf.JOINTS[name.split('_')[0]][name.split('_')[1]]
                    lm = lms[idx_joint] if lms is not None else None
                    row[name + "_x_px"] = lm.x * w if lm else None
                    row[name + "_y_px"] = lm.y * h if lm else None
                    row[name + "_visibility"] = lm.visibility if lm else None
                rows.append(row)
                if snapshots and usable:
                    snap = (idx, float(angle), frame.copy(), lms)
                    if first is None:
                        first = snap
                    if lowest is None or angle < lowest[1]:
                        lowest = snap
                    if highest is None or angle > highest[1]:
                        highest = snap
                    last = snap
                idx += 1
    finally:
        cap.release()
    if require_measured_time and len(times) != len(rows):
        raise ValueError(f'Measured frame times do not match decoded frames: {path}')
    if snapshots and first:
        shots = {"start": first, "lowest": lowest, "highest": highest, "end": last}
    result = summarize_frames(path, rows, settings,
                              'recorded_monotonic' if times is not None else 'estimated_from_nominal_fps',
                              'video_reprocessed')
    result.update(snapshots=shots, fps=fps, width=w, height=h)
    return result


def summarize_frames(path, rows, settings, timestamp_source, measurement_source):
    baseline_values = [r["shoulder_tilt_deg"] for r in rows if r["shoulder_tilt_deg"] is not None]
    baseline = statistics.median(baseline_values[:settings["tilt_baseline_frames"]]) if baseline_values else None
    for row in rows:
        row["tilt_change_deg"] = nf.tilt_change(row["shoulder_tilt_deg"], baseline)
    quality, ratio, bad_run = nf.frame_quality([r["usable"] for r in rows], settings)
    angles = [r["arm_elevation_deg"] for r in rows if r["usable"]]
    changes = [abs(r["tilt_change_deg"]) for r in rows if r["tilt_change_deg"] is not None]
    return {"path": str(path), "rows": rows, "angles": angles,
            "quality": quality, "usable_ratio": ratio, "max_bad_run": bad_run,
            "peak_tilt_change": max(changes) if changes else None,
            "baseline_tilt": baseline, "timestamp_source": timestamp_source,
            "measurement_source": measurement_source}


def save_frames_csv(result, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    if not rows:
        return
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[key for key in rows[0] if key != 'landmarks'], extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
