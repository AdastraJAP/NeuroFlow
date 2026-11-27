"""Record ONE shoulder-abduction repetition alone, with countdown + auto-stop.

Menu options 4/5/6/10 call this. Press SPACE once, get into position during
countdown, then recording starts/stops automatically. Q cancels.

Direct: python step4_record.py diagnostic D_01
        python step4_record.py reference ref_01
        python step4_record.py practice complete_01
        python step4_record.py test T_01
"""
import csv
import json
import re
import sys
import time
from contextlib import ExitStack
from datetime import datetime

import cv2
import nf_core as nf
from solo_timing import SoloTimer
from repetition import RepetitionDetector, RETURN_DELTA_DEG, RETURN_DWELL_MS
from bilateral import BilateralRepetitionDetector
from processing import (LIVE_POSE_SCHEMA, BILATERAL_LIVE_POSE_SCHEMA,
                        capture_landmarks, live_pose_path, measure_frame,
                        measure_bilateral_frame, save_live_pose)
from segmented_scoring import capture_segmentation, require_calibration


def overlay(frame, label, position, line2='', color=(255,255,255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 155), (35,35,35), -1)
    cv2.putText(frame, label, (18, 53), cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 3)
    cv2.putText(frame, line2, (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    cv2.putText(frame, f"{position.upper()} | shoulder abduction | q cancel", (18, 136),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ('diagnostic', 'reference', 'practice', 'test'):
        raise SystemExit('Usage: python step4_record.py [diagnostic|reference|practice|test] [clip_name]')
    category, name = sys.argv[1:]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise SystemExit('Clip name may contain only letters, digits, _ and -')

    s = nf.load_settings()
    if category == 'test' and s['arm'] != 'both':
        require_calibration(s)
    position = s['position']
    folder = nf.mode_dir('clips', s) / category
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + '.avi')
    ts_path = folder / (name + '.timestamps.csv')
    meta_path = folder / (name + '.json')
    if any(p.exists() for p in (path, ts_path, meta_path, live_pose_path(path))):
        raise SystemExit(f'Already exists: {path}. Use a different clip name.')

    countdown = s['record_countdown_s']
    duration_s = s['rep_timeout_s']
    timer = SoloTimer(countdown, duration_s=None)
    bilateral = s['arm'] == 'both'
    repetition = BilateralRepetitionDetector(duration_s) if bilateral else RepetitionDetector(duration_s)
    resources = ExitStack()
    cap, w, h = nf.open_camera(s.get('camera_index', 0), s['capture_width'], s['capture_height'])
    nominal_fps = cap.get(cv2.CAP_PROP_FPS)
    if not nominal_fps or not 1 <= nominal_fps <= 120:
        nominal_fps = 30.0
    window = f'Neuro-Flow {position} {s["arm"]} {category}/{name} | SPACE once | q cancel'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w,1280), min(h,720))
    writer, frames, times = None, 0, []
    live_rows = []
    result = None
    started_wallclock = None
    model_origin = model_last_ts = None
    print(f'Automatic repetition recorder | {s["version"]} | {position} | {path.resolve()}')
    print(f'Press SPACE once. Countdown {countdown}s, then automatic detection and stop.')
    print('Hold still until READY, then raise and lower once. Keep your upper arm in view.')
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print('Camera stopped unexpectedly. Incomplete clip will be discarded.')
                break
            now = time.monotonic()
            shown = cv2.flip(raw, 1)
            event = timer.tick(now)
            if event == 'started':
                if not bilateral:
                    detector = resources.enter_context(nf.make_landmarker())
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'),
                                         nominal_fps, (w,h))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create AVI. Check disk space and codec.')
                started_wallclock = datetime.now().isoformat(timespec='seconds')
                print('RECORDING STARTED automatically. Do one repetition.')
                if bilateral:
                    # This frame was read before AVI setup. Start the measured
                    # capture clock after setup and use the next camera frame.
                    timer.record_started_at = time.monotonic()
                    continue
            if timer.state == 'recording':
                timestamp = int(round(timer.elapsed(now)*1000))
                if bilateral and times:
                    timestamp = max(timestamp, times[-1][1] + 1)
                if bilateral:
                    model_timestamp = max(model_last_ts + 1, int(round((now - model_origin) * 1000)))
                    model_last_ts = model_timestamp
                else:
                    model_timestamp = max(timestamp, times[-1][1] + 1) if times else timestamp
                landmarks = nf.detect(detector, raw, model_timestamp)
                row = (measure_bilateral_frame if bilateral else measure_frame)(
                    frames, timestamp, landmarks, s, w, h)
                row['landmarks'] = capture_landmarks(landmarks)
                live_rows.append(row)
                writer.write(raw)
                times.append((frames, timestamp))
                frames += 1
                result = (repetition.push(timestamp, row['left_arm_elevation_deg'],
                                          row['right_arm_elevation_deg']) if bilateral else
                          repetition.push(timestamp, row['arm_elevation_deg']))
                row['detector_state'] = repetition.state
                if bilateral:
                    row['startup_gap_reset'] = repetition.startup_gap_reset
                    row['left_detector_state'] = repetition.arms['left'].state
                    row['right_detector_state'] = repetition.arms['right'].state
                if result is not None:
                    print(f'Automatic stop: {result.outcome} ({result.termination_reason})')
                    break
                if bilateral:
                    left = row['left_arm_elevation_deg']
                    right = row['right_arm_elevation_deg']
                    fmt = lambda value: f'{value:.0f}' if value is not None else '--'
                    if any(d.baseline is None for d in repetition.arms.values()):
                        prompt = 'WAIT: BOTH ARMS DOWN'
                    elif any(d.active for d in repetition.arms.values()):
                        prompt = 'MOVING: LOWER BOTH'
                    else:
                        prompt = 'BOTH READY - RAISE'
                    detail = (f'L {fmt(left)} {"OK" if row["left_usable"] else "LOST"}/{row["left_detector_state"][:5]} | '
                              f'R {fmt(right)} {"OK" if row["right_usable"] else "LOST"}/{row["right_detector_state"][:5]} | '
                              f'L-R {fmt(row["elevation_diff_signed_deg"])}')
                    overlay(shown, prompt, position,
                            detail,
                            (0, 255, 255) if prompt.startswith('WAIT') else (0, 255, 0))
                else:
                    overlay(shown, f'REC {timer.elapsed(now):.1f}s', position,
                            repetition.instruction, (0,0,255))
            elif timer.state == 'countdown':
                overlay(shown, f'GET READY: {timer.countdown_left(now)}', position,
                        'Hold both arms down until BOTH READY after recording begins.' if bilateral else
                        'Move into your marked position. Recording starts automatically.', (0,255,255))
            else:
                overlay(shown, 'READY - press SPACE once', position,
                        f'{countdown}s to prepare, up to {duration_s}s per repetition')
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('Cancelled. No incomplete clip will be kept.')
                break
            if key == ord(' ') and timer.state == 'ready':
                if bilateral:
                    # First VIDEO-mode inference is outside the saved capture.
                    # Keep its MediaPipe clock separate from measured frame time.
                    detector = resources.enter_context(nf.make_landmarker())
                    model_origin = time.monotonic()
                    nf.detect(detector, raw, 0)
                    model_last_ts = 0
                    print('Pose model warmed up before the five-second countdown.')
                timer.arm(time.monotonic())
                print(f'Countdown started ({countdown}s). Get into position.')
    finally:
        resources.close()
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

    if result is None or not frames:
        if path.exists():
            path.unlink()
        print('No completed recording saved.')
        return

    with ts_path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['frame','timestamp_ms'])
        wr.writerows(times)
    actual_span = (times[-1][1] - times[0][1]) / 1000 if frames > 1 else 0.0
    save_live_pose(path, live_rows)
    meta = {
        'clip': name, 'category': category, 'position': position,
        'version': s['version'], 'arm': s['arm'], 'reference_clip': s['reference_clip'],
        'angle_definition': s['angle_definition'], 'resolution': [w,h],
        'video_nominal_fps': nominal_fps, 'recording_countdown_s': countdown,
        'rep_timeout_s': duration_s, 'actual_duration_s': actual_span,
        'recording_schema_version': BILATERAL_LIVE_POSE_SCHEMA if bilateral else LIVE_POSE_SCHEMA,
        'capture_outcome': result.outcome, 'termination_elapsed_ms': times[-1][1],
        'capture_frames': frames, 'min_visibility': s['min_visibility'],
        'return_delta_deg': RETURN_DELTA_DEG, 'return_dwell_ms': RETURN_DWELL_MS,
        **(result.metadata() if bilateral else capture_segmentation(result)),
        'startup_gap_policy': 'one_rest_baseline_restart' if bilateral else None,
        'startup_gap_restarts': repetition.startup_gap_restarts if bilateral else None,
        'rest_baseline_policy': 'rolling_stable_window_500ms' if bilateral else None,
        'peak_policy': 'detected_movement_interval' if bilateral else None,
        'calibration_id': s['dtw_calibration']['id'] if category == 'test' and not bilateral else None,
        'actual_fps_approx': (frames-1)/actual_span if actual_span>0 else None,
        'timestamp_source': 'monotonic capture clock', 'record_started_at_local': started_wallclock,
        'camera_setup': s['camera_setup']
    }
    meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'Saved: {path} ({frames} frames, {actual_span:.2f}s between first/last frame)')
    print(f'Saved timestamps: {ts_path}')
    print(f'Saved capture pose: {live_pose_path(path)}')
    print(f'Saved metadata: {meta_path}')
    print('AVI playback FPS is nominal; recorded timestamps are the measured timing.')


if __name__ == '__main__':
    main()
