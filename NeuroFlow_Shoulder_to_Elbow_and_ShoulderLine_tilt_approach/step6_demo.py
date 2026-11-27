"""Hands-free Neuro-Flow demo: C/I/U label, SPACE once, countdown, auto-score.

Seated and standing use independent references/DTW thresholds and log files.
Manual start is only used to arm the countdown; no one needs to stop recording.
"""
import csv
import statistics
import time
from contextlib import ExitStack
from datetime import datetime

import cv2
import nf_core as nf
from solo_timing import SoloTimer
from repetition import RepetitionDetector, START_TIMEOUT_MS
from segmented_scoring import compare, load_reference, require_calibration


def append_log(row, settings):
    path = nf.mode_dir('logs', settings) / 'live_test_log.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            wr.writeheader()
        wr.writerow(row)
    return path


def score_rep(repetition, samples, tilts, baseline, expected, settings, ref):
    flags = [x[2] for x in samples]
    _, ratio, bad_run = nf.frame_quality(flags, settings)
    peak, score, message, color = compare(repetition, ref, settings)
    if baseline is None and tilts:
        baseline = statistics.median(tilts[:settings['tilt_baseline_frames']])
    changes = [abs(nf.tilt_change(t, baseline)) for t in tilts if baseline is not None]
    peak_tilt = max(changes) if changes else None
    if peak_tilt is not None and peak_tilt > settings['shoulder_tilt_warning_deg']:
        message += ' | shoulder-line tilt observed (exploratory)'
    duration = (samples[-1][0]-samples[0][0])/1000 if len(samples)>1 else 0.0
    row = {
        'time': datetime.now().isoformat(timespec='seconds'),
        'position': settings['position'],
        'expected_label_before_attempt': expected,
        'arm': settings['arm'], 'version': settings['version'],
        'calibration_id': settings['dtw_calibration']['id'],
        'repetition_outcome': repetition.outcome,
        'start_ms': repetition.start_ms,
        'end_ms': repetition.end_ms,
        'frames': len(samples), 'actual_duration_s': round(duration, 3),
        'usable_ratio': round(ratio,4), 'max_bad_run': bad_run,
        'max_arm_elevation_deg': round(peak,2) if peak is not None else '',
        'dtw_score': round(score,3) if score is not None else '',
        'peak_shoulder_tilt_change_deg': round(peak_tilt,2) if peak_tilt is not None else '',
        'system_result': message,
    }
    return row, message, color, score


def main():
    s = nf.load_settings()
    if s['arm'] == 'both':
        raise SystemExit('Bilateral scored demo is unavailable; use live measurement or capture.')
    require_calibration(s)
    ref = load_reference(s)
    print(f'Position: {s["position"]} | Segmented DTW: {s["dtw_threshold"]}')
    print('Choose C=complete, I=incomplete, U=uncertain BEFORE each attempt.')
    print(f'Then SPACE once -> {s["record_countdown_s"]}s countdown -> automatic detection and scoring.')
    print('Q quits. You never need to press SPACE to stop.')
    cap, w, h = nf.open_camera(s.get('camera_index',0), s['capture_width'], s['capture_height'])
    win = f'Neuro-Flow DEMO {s["position"]} | C/I/U label -> SPACE once | q quit'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w,1280), min(h,720))
    duration = START_TIMEOUT_MS / 1000 + s['rep_timeout_s'] + 2
    timer = SoloTimer(s['record_countdown_s'], duration)
    expected = None
    samples, tilts, baseline = [], [], None
    last_message, last_color, last_score = 'Select C/I/U, then SPACE to start countdown', (255,255,255), None
    last_log = None
    try:
        with ExitStack() as resources:
            while True:
                ok, raw = cap.read()
                if not ok:
                    print('Camera stopped. Any in-progress repetition was discarded.')
                    break
                now = time.monotonic()
                event = timer.tick(now)
                result = None
                lms, angle, usable, tilt = None, None, False, None
                if event == 'started':
                    samples, tilts, baseline = [], [], None
                    repetition = RepetitionDetector(s['rep_timeout_s'])
                    detector = resources.enter_context(nf.make_landmarker())
                    print('Recording started automatically. Do one repetition.')
                if timer.state == 'recording':
                    ts = int(round(timer.elapsed(now)*1000))
                    lms = nf.detect(detector, raw, max(ts, samples[-1][0] + 1) if samples else ts)
                    angle, usable = nf.shoulder_angle(lms, s['arm'], w, h, s['min_visibility'])
                    tilt = nf.shoulder_tilt(lms, w, h, s['min_visibility'])
                    samples.append((ts, angle if usable else None, bool(usable)))
                    result = repetition.push(ts, angle if usable else None)
                    if tilt is not None:
                        tilts.append(tilt)
                        if baseline is None and len(tilts) >= s['tilt_baseline_frames']:
                            baseline = statistics.median(tilts[:s['tilt_baseline_frames']])
                if event == 'finished' or result is not None:
                    row, last_message, last_color, last_score = score_rep(
                        repetition.finish(), samples, tilts, baseline, expected, s, ref)
                    last_log = append_log(row, s)
                    print(f'{s["position"]} attempt: {last_message} | DTW={last_score} | usable={row["usable_ratio"]:.0%}')
                    expected = None
                    resources.close()
                    timer = SoloTimer(s['record_countdown_s'], duration)

                shown = cv2.flip(raw, 1)
                if lms is not None:
                    nf.draw_skeleton(shown, lms, s['min_visibility'], mirror=True)
                    nf.highlight_joints(shown, lms, s['arm'], mirror=True)
                cv2.putText(shown, f'{s["position"].upper()} | {s["arm"]} arm: {angle:.1f} deg' if usable else
                            f'{s["position"].upper()} | {"Arm joints unreliable" if timer.state == "recording" else "Live attempt idle"}', (15,38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0,255,0) if usable else (0,165,255), 2)
                delta = nf.tilt_change(tilt, baseline)
                cv2.putText(shown, f'Shoulder-line change: {delta:+.1f} deg' if delta is not None else
                            'Shoulder-line change: baseline pending', (15,72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255,255,255), 2)
                cv2.putText(shown, f'Target {s["movement_target_deg"]:.0f} deg | DTW: {"calibrated" if s["dtw_threshold"] is not None else "not calibrated"}',
                            (15,104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                cv2.putText(shown, f'Expected BEFORE attempt: {expected or "SELECT C/I/U"}',
                            (15,136), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                if timer.state == 'countdown':
                    cv2.putText(shown, f'GET READY: {timer.countdown_left(now)}',
                                (15,192), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
                elif timer.state == 'recording':
                    cv2.putText(shown, repetition.instruction,
                                (15,192), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                else:
                    cv2.putText(shown, 'C/I/U label -> SPACE once -> walk into position',
                                (15,192), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255,255,255), 2)
                if last_score is not None:
                    cv2.putText(shown, f'Last DTW: {last_score:.2f}', (15,224),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                cv2.rectangle(shown, (0,h-60), (w,h), (35,35,35), -1)
                cv2.putText(shown, last_message[:85], (12,h-22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, last_color, 2)
                cv2.imshow(win, shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('Quit. Any in-progress attempt was discarded.')
                    break
                if timer.state == 'ready' and key in (ord('c'), ord('i'), ord('u')):
                    expected = {ord('c'):'complete', ord('i'):'incomplete', ord('u'):'uncertain'}[key]
                    print('Expected label set BEFORE attempt:', expected)
                if timer.state == 'ready' and key == ord(' '):
                    if expected is None:
                        print('First select expected label: C, I or U.')
                        last_message = 'Select C/I/U first, then press SPACE once'
                    else:
                        timer.arm(time.monotonic())
                        print(f'Countdown started ({s["record_countdown_s"]}s). Get into position.')
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if last_log:
        print('Saved results:', last_log)


if __name__ == '__main__':
    main()
