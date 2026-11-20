"""Shared inputs and outcomes for offline and live comparison."""
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path

import nf_core as nf
from processing import LIVE_POSE_SCHEMA, live_pose_path, load_live_pose, process_clip
from repetition import detect_repetition, resample


def scoring_dir(settings):
    return nf.mode_dir('traces', settings) / 'capture_pose_v1'


def reference_identity(settings):
    reference = Path(settings['reference_clip'])
    with reference.open('rb') as video, reference.with_suffix('.timestamps.csv').open('rb') as timestamps:
        identity = {
            'reference_clip': reference.as_posix(),
            'video_sha256': hashlib.file_digest(video, 'sha256').hexdigest(),
            'timestamps_sha256': hashlib.file_digest(timestamps, 'sha256').hexdigest(),
            **{key: settings[key] for key in (
                'version', 'position', 'arm', 'min_visibility', 'rep_timeout_s',
                'movement_target_deg', 'target_tolerance_deg',
            )},
        }
    metadata = json.loads(reference.with_suffix('.json').read_text(encoding='utf-8'))
    if metadata.get('recording_schema_version', 0) >= LIVE_POSE_SCHEMA:
        with live_pose_path(reference).open('rb') as pose:
            identity['pose_sha256'] = hashlib.file_digest(pose, 'sha256').hexdigest()
        identity['measurement_source'] = 'live_capture'
    else:
        identity['measurement_source'] = 'video_reprocessed'
    return identity


def require_calibration(settings):
    threshold = settings['dtw_threshold']
    if (threshold is None or not isfinite(threshold) or not settings['reference_clip']
            or not settings.get('dtw_calibration')):
        raise SystemExit('Segmented DTW needs fresh practice calibration. Use menu options 5, 6, 8 and 9.')
    try:
        calibration = settings['dtw_calibration']
        matches = (calibration['threshold'] == threshold and bool(calibration['id'])
                   and calibration['reference'] == reference_identity(settings))
    except (OSError, ValueError, KeyError, TypeError):
        matches = False
    if not matches:
        raise SystemExit('Reference or scoring settings changed. Re-score fresh practice and freeze a threshold.')


def load_reference(settings):
    trace_dir = scoring_dir(settings)
    try:
        metadata = json.loads((trace_dir / 'reference_meta.json').read_text(encoding='utf-8'))
        if metadata != reference_identity(settings):
            raise ValueError('reference changed')
        with (trace_dir / 'reference.csv').open(newline='', encoding='utf-8') as source:
            samples = [(int(row['timestamp_ms']), float(row['angle_deg'])) for row in csv.DictReader(source)]
        return resample(samples)
    except (OSError, ValueError, KeyError):
        raise SystemExit('Current segmented reference trace is unavailable. Run menu option 8.') from None


def detect_clip(result, settings):
    samples = ((row['timestamp_ms'], row['arm_elevation_deg'] if row['usable'] else None)
               for row in result['rows'])
    return detect_repetition(samples, settings['rep_timeout_s'])


def capture_segmentation(repetition):
    duration = ((repetition.end_ms - repetition.start_ms) / 1000
                if repetition.end_ms is not None else None)
    return {
        'detector_status': repetition.outcome, 'termination_reason': repetition.termination_reason,
        'repetition_start_ms': repetition.start_ms, 'repetition_end_ms': repetition.end_ms,
        'repetition_duration_s': duration, 'rest_baseline_deg': repetition.rest_baseline_deg,
    }


def scoring_measurements(path, settings, metadata, *, legacy_reference=False):
    if legacy_reference and metadata.get('recording_schema_version', 0) < LIVE_POSE_SCHEMA:
        result = process_clip(path, settings, require_measured_time=True)
        return result, detect_clip(result, settings)
    result = load_live_pose(path, settings, metadata)
    repetition = detect_clip(result, settings)
    if any(metadata.get(key) != value for key, value in capture_segmentation(repetition).items()):
        raise ValueError(f'{path.name}: live pose replay disagrees with recorded segmentation metadata')
    return result, repetition


def compare(repetition, reference, settings):
    if repetition.outcome == 'unable_to_judge':
        return None, None, 'Unable to judge - tracking or resting baseline unavailable', (0, 165, 255)
    if repetition.outcome == 'no_start':
        return None, None, 'No repetition start detected', (0, 165, 255)
    if repetition.outcome == 'incomplete_return':
        return None, None, 'Incomplete return - arm did not settle at rest', (0, 0, 255)
    peak = max(angle for _, angle in repetition.segment)
    score = nf.dtw_distance(reference, resample(repetition.segment))[0]
    message, color = nf.decide(peak, score, settings, False)
    return peak, score, message, color
