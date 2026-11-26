"""Score timestamped repetitions and freeze a threshold from fresh practice."""
import csv
import json
import math
import sys
from pathlib import Path, PureWindowsPath
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nf_core as nf
from config import LEGACY_VERSION
from processing import LIVE_POSE_SCHEMA, save_frames_csv
from repetition import Repetition, resample
from segmented_scoring import compare, reference_identity, require_calibration, scoring_dir, scoring_measurements


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def practice_bounds(rows, settings):
    target_min = settings['movement_target_deg'] - settings['target_tolerance_deg']

    def scores(prefix):
        return [float(row['dtw_score']) for row in rows
                if row['clip'].startswith(prefix + '_') and row['dtw_score'] not in (None, '')
                and (prefix != 'complete' or float(row['max_arm_elevation_deg']) >= target_min)]

    good, bad = scores('complete'), scores('incomplete')
    if len(good) >= 3 and len(bad) >= 3 and max(good) < min(bad):
        return max(good), min(bad)
    return None


def freeze(value, settings, traces, identity):
    try:
        metadata = json.loads((traces / 'practice_meta.json').read_text(encoding='utf-8'))
        with (traces / 'summary_practice.csv').open(newline='', encoding='utf-8') as source:
            bounds = practice_bounds(list(csv.DictReader(source)), settings)
        threshold = float(value)
    except (OSError, ValueError, KeyError):
        raise SystemExit('Run option 8 and enter a numeric practice threshold.') from None
    if metadata != identity or bounds is None:
        raise SystemExit('Need current practice results and at least 3+3 scored examples with separated DTW scores.')
    if not math.isfinite(threshold) or not bounds[0] < threshold < bounds[1]:
        raise SystemExit(f'Threshold must be strictly between {bounds[0]:.8g} and {bounds[1]:.8g}.')
    if settings.get('dtw_calibration', {}) and settings['dtw_calibration']['reference'] == identity:
        raise SystemExit('A threshold is already frozen for this reference and setup. Use a new reference for a new evaluation.')
    settings['dtw_threshold'] = threshold
    settings['dtw_calibration'] = {'reference': identity, 'id': uuid4().hex, 'threshold': threshold}
    nf.save_settings(settings)
    print(f'Frozen segmented DTW threshold: {threshold:.8g}. Record new held-out tests with option 10.')


def clip_metadata(path):
    try:
        return json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def plot_trace(axis, result, repetition, label):
    if result is None:
        return
    axis.plot([row['timestamp_ms'] / 1000 for row in result['rows']],
              [row['arm_elevation_deg'] for row in result['rows']], alpha=0.35)
    if repetition.segment:
        timestamps, angles = zip(*repetition.segment)
        axis.plot([timestamp / 1000 for timestamp in timestamps], angles, label=label)


def main():
    args = sys.argv[1:]
    which = args[0] if args else 'practice'
    if not (which in ('practice', 'test') and len(args) <= 1 or which == '--freeze' and len(args) == 2):
        raise SystemExit('Use step5_score.py [practice|test] or step5_score.py --freeze THRESHOLD.')
    settings = nf.load_settings()
    if which == 'test':
        require_calibration(settings)
    if not settings['reference_clip']:
        raise SystemExit('Record a reference with menu option 4.')
    path = Path(settings['reference_clip'])
    metadata = clip_metadata(path)
    if (metadata.get('category') != 'reference' or metadata.get('position') != settings['position']
            or metadata.get('arm') != settings['arm']
            or metadata.get('version') not in (LEGACY_VERSION, settings['version'])):
        raise SystemExit('Reference metadata does not match this arm and position. Record a new reference with option 4.')
    try:
        identity = reference_identity(settings)
    except (OSError, ValueError):
        raise SystemExit('Reference video, capture data or metadata unavailable. Use option 4 to record a reference.') from None
    traces = scoring_dir(settings)
    traces.mkdir(parents=True, exist_ok=True)
    if which == '--freeze':
        freeze(args[1], settings, traces, identity)
        return
    practice_meta = traces / 'practice_meta.json'
    if which == 'practice':
        practice_meta.unlink(missing_ok=True)
        settings['dtw_threshold'] = None
    try:
        ref, rep = scoring_measurements(path, settings, metadata, legacy_reference=True)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        raise SystemExit(str(error)) from None
    print(f'Reference measurement source: {ref["measurement_source"]}')
    if rep.outcome != 'complete':
        raise SystemExit(f'Reference detection: {rep.outcome}. Record a full return with option 4.')
    reference = resample(rep.segment)
    save_frames_csv(ref, traces / 'reference_frames.csv')
    write_csv(traces / 'reference.csv', [
        {'timestamp_ms': timestamp, 'angle_deg': angle} for timestamp, angle in rep.segment
    ])
    (traces / 'reference_meta.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    folder = nf.mode_dir('clips', settings) / which
    labels = {}
    if which == 'test' and (folder / 'expected_labels.csv').exists():
        with (folder / 'expected_labels.csv').open(newline='', encoding='utf-8') as source:
            labels = {row['clip']: row['expected_label'] for row in csv.DictReader(source)}
    summary = []
    figure, axis = plt.subplots(figsize=(10, 5))
    plot_trace(axis, ref, rep, 'reference')
    for clip in sorted(folder.glob('*.avi')):
        metadata = clip_metadata(clip)
        if (any(metadata.get(key) != settings[key] for key in ('version', 'position', 'arm', 'rep_timeout_s'))
                or metadata.get('category') != which
                or PureWindowsPath(metadata.get('reference_clip') or '') != PureWindowsPath(path)
                or (which == 'test' and metadata.get('calibration_id') != settings['dtw_calibration']['id'])):
            print(f'Skipping incompatible recording: {clip.name}')
            continue
        result = None
        measurement_source = 'unavailable'
        try:
            result, repetition = scoring_measurements(clip, settings, metadata)
            measurement_source = result['measurement_source']
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            reason = ('legacy_pose_unavailable' if metadata.get('recording_schema_version', 0) < LIVE_POSE_SCHEMA
                      else 'live_pose_invalid')
            repetition = Repetition('unable_to_judge', termination_reason=reason)
            print(error)
        if result is not None:
            save_frames_csv(result, traces / f'{which}_{clip.stem}.csv')
        peak, score, message, _ = compare(repetition, reference, settings)
        if result is None:
            message = 'Unable to judge - exact capture measurements unavailable or inconsistent'
        tilt = result['peak_tilt_change'] if result is not None else None
        if tilt is not None and tilt > settings['shoulder_tilt_warning_deg']:
            message += ' | shoulder-line tilt observed (exploratory)'
        summary.append({
            'clip': clip.stem, 'expected_label': labels.get(clip.stem, ''),
            'measurement_source': measurement_source, 'capture_outcome': metadata.get('capture_outcome'),
            'repetition_outcome': repetition.outcome, 'start_ms': repetition.start_ms,
            'end_ms': repetition.end_ms, 'max_arm_elevation_deg': peak, 'dtw_score': score,
            'rest_baseline_deg': repetition.rest_baseline_deg, 'termination_reason': repetition.termination_reason,
            'usable_ratio': result['usable_ratio'] if result is not None else None,
            'max_bad_run': result['max_bad_run'] if result is not None else None,
            'peak_tilt_change_deg': tilt, 'system_result': message,
        })
        print(f'{clip.stem}: {message} | DTW={score}')
        plot_trace(axis, result, repetition, clip.stem)
    if not summary:
        raise SystemExit(f'No compatible {which} recordings. Record fresh attempts for this setup.')
    axis.set(xlabel='Measured time (s)', ylabel='Arm elevation (degrees)', title=f'{which.title()} repetitions')
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(traces / f'angles_{which}.png', dpi=140)
    plt.close(figure)
    write_csv(traces / f'summary_{which}.csv', summary)
    if which == 'practice':
        practice_meta.write_text(json.dumps(identity, indent=2), encoding='utf-8')
        bounds = practice_bounds(summary, settings)
        if bounds:
            print(f'Suggested practice threshold for option 9: {sum(bounds) / 2:.12g}')
        else:
            print('Need 3+3 scoreable practice clips with separated DTW scores.')
            print('For incomplete practice that can calibrate DTW, make a shorter or different raise and still return to rest.')
    else:
        print('Report every held-out attempt, including abstentions. Keep the practice threshold fixed.')
    print(f'Saved traces, plot, and summary in {traces}')


if __name__ == '__main__':
    main()
