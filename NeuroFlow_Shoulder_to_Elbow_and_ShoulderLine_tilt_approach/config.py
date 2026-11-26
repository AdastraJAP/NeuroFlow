"""Experiment configuration. Seated and standing calibrations are independent.

No camera/MediaPipe imports: configuration can be tested without hardware.
"""
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / 'settings.json'
POSITIONS = ('seated', 'standing')
LEGACY_VERSION = 'hip_free_2d_v3_solo_positions'
VERSION = 'hip_free_2d_v4_segmented'

DEFAULT_SETTINGS = {
    'version': VERSION,
    'exercise': 'shoulder_abduction',
    'arm': 'right',
    'position': 'seated',
    'angle_definition': 'shoulder_to_elbow_vs_downward_image_vertical',
    'coordinate_representation': 'image_landmarks_converted_to_pixels',
    'camera_setup': 'fixed frontal view, camera at shoulder/chest height, marked position, full upper arm visible',
    'capture_width': 1280,
    'capture_height': 720,
    'camera_index': 0,
    'movement_target_deg': 90.0,
    'target_tolerance_deg': 10.0,
    'min_visibility': 0.5,
    'min_usable_ratio': 0.9,
    'max_bad_run': 4,
    'tilt_baseline_frames': 8,
    'shoulder_tilt_warning_deg': 10.0,
    'record_countdown_s': 5,
    'record_duration_s': 10,
    'rep_timeout_s': 25,
    'conditions': {
        'seated': {'reference_clip': None, 'dtw_threshold': None},
        'standing': {'reference_clip': None, 'dtw_threshold': None},
    },
}


def _read(path):
    path = Path(path)
    data = deepcopy(DEFAULT_SETTINGS)
    if path.exists():
        with path.open(encoding='utf-8') as f:
            raw = json.load(f)
        if raw.get('version') not in (VERSION, LEGACY_VERSION):
            raise SystemExit('Settings are from an unsupported project version.')
        data.update({k: v for k, v in raw.items() if k != 'conditions'})
        for position in POSITIONS:
            data['conditions'][position].update(raw.get('conditions', {}).get(position, {}))
        if raw['version'] == LEGACY_VERSION:
            data['version'] = VERSION
            for condition in data['conditions'].values():
                condition['dtw_threshold'] = None
    if data['position'] not in POSITIONS:
        raise SystemExit(f"Invalid position: {data['position']}. Choose seated or standing.")
    if data['arm'] not in ('left', 'right'):
        raise SystemExit('Arm must be left or right.')
    return data


def _write(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def load_settings(path=SETTINGS_PATH):
    data = _read(path)
    active = data['position']
    flat = {k: v for k, v in data.items() if k != 'conditions'}
    flat.update(data['conditions'][active])
    return flat


def save_settings(settings, path=SETTINGS_PATH):
    data = _read(path)
    active = settings.get('position', data['position'])
    if active not in POSITIONS:
        raise ValueError('position must be seated or standing')
    for key in DEFAULT_SETTINGS:
        if key != 'conditions' and key in settings:
            data[key] = settings[key]
    for key in ('reference_clip', 'dtw_threshold', 'dtw_calibration'):
        if key in settings:
            data['conditions'][active][key] = settings[key]
    _write(data, path)


def set_position(position, path=SETTINGS_PATH):
    if position not in POSITIONS:
        raise ValueError('position must be seated or standing')
    data = _read(path)
    data['position'] = position
    _write(data, path)
    return load_settings(path)


def set_recording_timing(countdown, duration, path=SETTINGS_PATH):
    countdown, duration = int(countdown), int(duration)
    if not 3 <= countdown <= 20:
        raise ValueError('Countdown must be 3-20 seconds.')
    if not 6 <= duration <= 30:
        raise ValueError('Recording must be 6-30 seconds.')
    data = _read(path)
    data['record_countdown_s'] = countdown
    data['rep_timeout_s'] = duration
    _write(data, path)
    return load_settings(path)


def mode_dir(kind, settings=None):
    s = settings if settings is not None else load_settings()
    return Path(kind) / s['position'] / 'segmented_v1'
