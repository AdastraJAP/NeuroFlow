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
BILATERAL_LIVE_POSE_SCHEMA = 6


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


def measure_bilateral_frame(index, timestamp, landmarks, settings, width, height):
    """Measure anatomical arms independently from one pose and capture clock.

    Signed difference is left minus right. No single-arm angle is assigned to
    ``arm_elevation_deg`` in both mode, preserving its unilateral meaning.
    """
    row = {'frame': index, 'timestamp_ms': timestamp, 'arm': 'both',
           'angle_definition': settings['angle_definition'],
           'pose_detected': int(landmarks is not None),
           'arm_elevation_deg': None, 'usable': 0,
           'shoulder_tilt_deg': nf.shoulder_tilt(
               landmarks, width, height, settings['min_visibility'])}
    for arm in ('left', 'right'):
        angle, valid = nf.shoulder_angle(
            landmarks, arm, width, height, settings['min_visibility'])
        row[f'{arm}_arm_elevation_deg'] = angle if valid else None
        row[f'{arm}_usable'] = int(valid)
        row[f'{arm}_status'] = ('ok' if valid else
                                'no_person' if landmarks is None else 'unreliable_arm_joints')
        for joint in ('shoulder', 'elbow'):
            lm = (landmarks[nf.JOINTS[arm][joint]] if landmarks is not None else None)
            prefix = f'{arm}_{joint}'
            row[f'{prefix}_x_px'] = lm.x * width if lm is not None and math.isfinite(lm.x) else None
            row[f'{prefix}_y_px'] = lm.y * height if lm is not None and math.isfinite(lm.y) else None
            row[f'{prefix}_visibility'] = (lm.visibility if lm is not None and
                                            math.isfinite(lm.visibility) else None)
            row[f'{prefix}_presence'] = (lm.presence if lm is not None and
                                          math.isfinite(lm.presence) else None)
    left, right = row['left_arm_elevation_deg'], row['right_arm_elevation_deg']
    row['elevation_diff_signed_deg'] = left - right if left is not None and right is not None else None
    row['elevation_diff_abs_deg'] = abs(row['elevation_diff_signed_deg']) if row['elevation_diff_signed_deg'] is not None else None
    row['status'] = ('both_valid' if row['left_usable'] and row['right_usable'] else
                     'partial_tracking' if row['left_usable'] or row['right_usable'] else
                     'no_person' if landmarks is None else 'both_unreliable')
    return row


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
    if settings['arm'] == 'both':
        return load_bilateral_live_pose(path, settings, metadata)
    if metadata.get('recording_schema_version', 0) < LIVE_POSE_SCHEMA:
        raise ValueError(f'{path.name}: exact live pose was not saved by this recorder')
    if metadata.get('min_visibility') != settings['min_visibility']:
        raise ValueError(f'{path.name}: recorded visibility setting differs from the current setup')
    if metadata.get('arm') is not None and metadata['arm'] != settings['arm']:
        raise ValueError(f'{path.name}: recorded arm differs from the current setup')
    with live_pose_path(path).open(encoding='utf-8') as source:
        rows = [json.loads(line) for line in source]
    times = nf.clip_timestamps(path, None)
    if (not rows or len(rows) != metadata.get('capture_frames')
            or [row['timestamp_ms'] for row in rows] != times
            or rows[-1]['timestamp_ms'] != metadata.get('termination_elapsed_ms')):
        raise ValueError(f'{path.name}: live pose does not match the capture timestamps and frame count')
    for index, row in enumerate(rows):
        angle = row['arm_elevation_deg']
        if (row['frame'] != index or type(row['timestamp_ms']) is not int or row['usable'] not in (0, 1)
                or (row.get('arm') is not None and row['arm'] != settings['arm'])
                or bool(row['usable']) != (angle is not None)
                or (angle is not None and not math.isfinite(angle))
                or 'landmarks' not in row):
            raise ValueError(f'{path.name}: invalid live pose row {index}')
    return summarize_frames(path, rows, settings, 'recorded_monotonic', 'live_capture')


def load_bilateral_live_pose(path, settings, metadata):
    """Replay schema-4/5/6 synchronized rows without rerunning MediaPipe."""
    from bilateral import BilateralRepetitionDetector

    path = Path(path)
    schema = metadata.get('recording_schema_version')
    if schema not in (4, 5, BILATERAL_LIVE_POSE_SCHEMA) or metadata.get('arm') != 'both':
        raise ValueError(f'{path.name}: not a bilateral live-pose recording')
    if metadata.get('min_visibility') != settings['min_visibility']:
        raise ValueError(f'{path.name}: recorded visibility setting differs from current setup')
    if metadata.get('angle_definition') != settings['angle_definition']:
        raise ValueError(f'{path.name}: recorded angle definition differs from current setup')
    with live_pose_path(path).open(encoding='utf-8') as source:
        rows = [json.loads(line) for line in source]
    with path.with_suffix('.timestamps.csv').open(newline='', encoding='utf-8') as source:
        stamp_rows = list(csv.DictReader(source))
    try:
        indices = [int(row['frame']) for row in stamp_rows]
        times = [int(row['timestamp_ms']) for row in stamp_rows]
    except (ValueError, KeyError) as exc:
        raise ValueError(f'{path.name}: invalid bilateral timestamp CSV') from exc
    if indices != list(range(len(stamp_rows))):
        raise ValueError(f'{path.name}: noncontiguous bilateral timestamp indices')
    if (not rows or len(rows) != metadata.get('capture_frames') or times is None
            or len(times) != len(rows) or rows[-1]['timestamp_ms'] != metadata.get('termination_elapsed_ms')):
        raise ValueError(f'{path.name}: live pose does not match capture count and time')
    detector = BilateralRepetitionDetector(metadata['rep_timeout_s'],
                                            restart_startup_gap=schema >= 5,
                                            reliable_baseline=schema >= 6)
    prior = -1
    for index, (row, timestamp) in enumerate(zip(rows, times)):
        if (row.get('frame') != index or type(timestamp) is not int
                or row.get('timestamp_ms') != timestamp or timestamp <= prior
                or row.get('arm') != 'both' or row.get('angle_definition') != settings['angle_definition']
                or 'landmarks' not in row):
            raise ValueError(f'{path.name}: invalid bilateral row {index}')
        prior = timestamp
        for arm in ('left', 'right'):
            angle, valid = row.get(f'{arm}_arm_elevation_deg'), row.get(f'{arm}_usable')
            if valid not in (0, 1) or bool(valid) != (angle is not None) or (
                    angle is not None and (not isinstance(angle, (int, float)) or not math.isfinite(angle))):
                raise ValueError(f'{path.name}: invalid {arm} measurement at frame {index}')
        left, right = row['left_arm_elevation_deg'], row['right_arm_elevation_deg']
        diff = left - right if left is not None and right is not None else None
        if row.get('elevation_diff_signed_deg') != diff or row.get('elevation_diff_abs_deg') != (abs(diff) if diff is not None else None):
            raise ValueError(f'{path.name}: invalid bilateral difference at frame {index}')
        result = detector.push(timestamp, left, right)
        if schema >= 5 and row.get('startup_gap_reset') is not detector.startup_gap_reset:
            raise ValueError(f'{path.name}: startup-gap decision disagrees at frame {index}')
        if row.get('detector_state') != detector.state:
            raise ValueError(f'{path.name}: detector replay disagrees at frame {index}')
        if result and index != len(rows)-1:
            raise ValueError(f'{path.name}: frames follow automatic detector stop')
    result = detector.result or detector.finish()
    if schema >= 5 and (metadata.get('startup_gap_policy') != 'one_rest_baseline_restart'
                        or metadata.get('startup_gap_restarts') != detector.startup_gap_restarts):
        raise ValueError(f'{path.name}: startup-gap metadata disagrees with replay')
    if schema >= 6 and (metadata.get('rest_baseline_policy') != 'rolling_stable_window_500ms'
                        or metadata.get('peak_policy') != 'detected_movement_interval'):
        raise ValueError(f'{path.name}: bilateral reliability policy differs from replay')
    if result.metadata() != {key: metadata.get(key) for key in result.metadata()}:
        raise ValueError(f'{path.name}: bilateral replay disagrees with metadata')
    return {'path': str(path), 'rows': rows, 'timestamp_source': 'recorded_monotonic',
            'measurement_source': 'live_capture', 'repetition': result}


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
