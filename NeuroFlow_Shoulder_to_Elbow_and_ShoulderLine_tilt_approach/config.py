"""Experiment configuration. Position and anatomical arm calibrations are independent.

No camera/MediaPipe imports: configuration can be tested without hardware.
"""
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / 'settings.json'
POSITIONS = ('seated', 'standing')
ARMS = ('right', 'left', 'both')
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
    # Existing conditions and paths remain the authoritative right-arm record.
    # New arm conditions are separate so old right-arm settings need no migration.
    'arm_conditions': {
        'left': {
            'seated': {'reference_clip': None, 'dtw_threshold': None, 'dtw_calibration': None},
            'standing': {'reference_clip': None, 'dtw_threshold': None, 'dtw_calibration': None},
        },
        # Bilateral capture is deliberately uncalibrated; it has no DTW scorer.
        'both': {
            'seated': {'reference_clip': None, 'dtw_threshold': None, 'dtw_calibration': None},
            'standing': {'reference_clip': None, 'dtw_threshold': None, 'dtw_calibration': None},
        },
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
        data.update({k: v for k, v in raw.items() if k not in ('conditions', 'arm_conditions')})
        for position in POSITIONS:
            data['conditions'][position].update(raw.get('conditions', {}).get(position, {}))
        for arm, conditions in raw.get('arm_conditions', {}).items():
            if arm not in data['arm_conditions']:
                raise SystemExit(f'Unsupported arm condition: {arm}')
            for position in POSITIONS:
                data['arm_conditions'][arm][position].update(conditions.get(position, {}))
        if raw['version'] == LEGACY_VERSION:
            data['version'] = VERSION
            for condition in data['conditions'].values():
                condition['dtw_threshold'] = None
    if data['position'] not in POSITIONS:
        raise SystemExit(f"Invalid position: {data['position']}. Choose seated or standing.")
    if data['arm'] not in ARMS:
        raise SystemExit('Arm must be left, right, or both.')
    if data['arm'] == 'both' and data['position'] != 'seated':
        raise SystemExit('Bilateral capture is currently supported only while seated.')
    if any(data['arm_conditions']['both']['seated'].get(key) is not None
           for key in ('dtw_threshold', 'dtw_calibration')):
        raise SystemExit('Bilateral DTW calibration is unsupported; remove it from local settings.')
    return data


def _write(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def load_settings(path=SETTINGS_PATH):
    data = _read(path)
    flat = {k: v for k, v in data.items() if k not in ('conditions', 'arm_conditions')}
    flat.update(_condition(data, data['arm'], data['position']))
    return flat


def _condition(data, arm, position):
    return data['conditions'][position] if arm == 'right' else data['arm_conditions'][arm][position]


def save_settings(settings, path=SETTINGS_PATH):
    data = _read(path)
    active = settings.get('position', data['position'])
    if active not in POSITIONS:
        raise ValueError('position must be seated or standing')
    arm = settings.get('arm', data['arm'])
    if arm not in ARMS:
        raise ValueError('arm must be left, right, or both')
    if arm == 'both' and active != 'seated':
        raise ValueError('bilateral capture is currently seated only')
    if arm == 'both' and any(settings.get(key) is not None
                             for key in ('dtw_threshold', 'dtw_calibration')):
        raise ValueError('bilateral DTW calibration is unsupported')
    for key in DEFAULT_SETTINGS:
        if key not in ('conditions', 'arm_conditions') and key in settings:
            data[key] = settings[key]
    for key in ('reference_clip', 'dtw_threshold', 'dtw_calibration'):
        if key in settings:
            _condition(data, arm, active)[key] = settings[key]
    _write(data, path)


def set_position(position, path=SETTINGS_PATH):
    if position not in POSITIONS:
        raise ValueError('position must be seated or standing')
    data = _read(path)
    if data['arm'] == 'both' and position != 'seated':
        raise ValueError('bilateral capture is currently seated only; select one arm first')
    data['position'] = position
    _write(data, path)
    return load_settings(path)


def set_arm(arm, path=SETTINGS_PATH):
    """Select anatomical arm without copying its reference or calibration."""
    if arm not in ARMS:
        raise ValueError('arm must be left, right, or both')
    data = _read(path)
    if arm == 'both' and data['position'] != 'seated':
        raise ValueError('bilateral capture is currently seated only')
    data['arm'] = arm
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
    base = Path(kind) / s['position']
    return (base / 'segmented_v1' if s['arm'] == 'right'
            else base / s['arm'] / 'segmented_v1')
