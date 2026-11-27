"""A6 geometry, capture state, isolation, and exact pose replay."""
import csv
import io
import json
import sys
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import menu
import nf_core as nf
import step4_record
from bilateral import BilateralRepetitionDetector
from config import load_settings, mode_dir, save_settings, set_arm, set_position
from processing import load_bilateral_live_pose, measure_bilateral_frame


def pose(left=0, right=0):
    points = [SimpleNamespace(x=0.5, y=1.5, z=0.0, visibility=0.0, presence=0.0)
              for _ in range(33)]
    for arm, x, angle in (('left', 0.3, left), ('right', 0.7, right)):
        shoulder = nf.JOINTS[arm]['shoulder']
        elbow = nf.JOINTS[arm]['elbow']
        points[shoulder] = SimpleNamespace(x=x, y=0.3, z=0.0, visibility=.99, presence=.99)
        dx = -0.3 if arm == 'left' else 0.3
        points[elbow] = SimpleNamespace(x=x + dx * angle / 90, y=0.6 - 0.3 * angle / 90,
                                        z=0.0, visibility=.99, presence=.99)
    return points


def cycle(left_start=600, right_start=600, left_return=2000, right_return=2000,
          missing=None, until=3000):
    detector = BilateralRepetitionDetector(25)
    rows = []
    for t in range(0, until + 1, 100):
        left = 5 if t < left_start or (left_return is not None and t >= left_return) else 90
        right = 5 if t < right_start or (right_return is not None and t >= right_return) else 90
        if missing and missing[0] <= t <= missing[1]:
            if missing[2] in ('left', 'both'): left = None
            if missing[2] in ('right', 'both'): right = None
        rows.append((t, left, right))
        result = detector.push(t, left, right)
        if result:
            return detector, result, rows
    return detector, detector.finish(), rows


class TestBilateralGeometry(unittest.TestCase):
    def setUp(self):
        self.settings = {'min_visibility': .5,
                         'angle_definition': 'shoulder_to_elbow_vs_downward_image_vertical'}

    def test_both_angles_and_difference_use_only_anatomical_shoulders_elbows(self):
        points = pose(left=90, right=45)
        row = measure_bilateral_frame(7, 123, points, self.settings, 1000, 1000)
        self.assertEqual((row['frame'], row['timestamp_ms'], row['arm']), (7, 123, 'both'))
        self.assertAlmostEqual(row['left_arm_elevation_deg'], 90)
        self.assertAlmostEqual(row['right_arm_elevation_deg'], 45)
        self.assertAlmostEqual(row['elevation_diff_signed_deg'], 45)
        self.assertAlmostEqual(row['elevation_diff_abs_deg'], 45)
        self.assertEqual((row['left_usable'], row['right_usable']), (1, 1))
        self.assertIsNone(row['arm_elevation_deg'])
        self.assertEqual(points[nf.JOINTS['left']['hip']].visibility, 0)
        self.assertEqual(points[nf.JOINTS['right']['wrist']].visibility, 0)
        self.assertEqual(row['left_shoulder_x_px'], 300)
        self.assertEqual(row['right_shoulder_x_px'], 700)

    def test_mirror_and_independent_missing(self):
        points = pose(45, 45)
        original = measure_bilateral_frame(0, 0, points, self.settings, 1000, 1000)
        for point in points:
            point.x = 1 - point.x
        mirrored = measure_bilateral_frame(0, 0, points, self.settings, 1000, 1000)
        self.assertAlmostEqual(original['left_arm_elevation_deg'], mirrored['left_arm_elevation_deg'])
        self.assertAlmostEqual(original['right_arm_elevation_deg'], mirrored['right_arm_elevation_deg'])
        points[nf.JOINTS['right']['elbow']].visibility = 0
        row = measure_bilateral_frame(1, 100, points, self.settings, 1000, 1000)
        self.assertEqual((row['left_usable'], row['right_usable']), (1, 0))
        self.assertIsNotNone(row['left_arm_elevation_deg'])
        self.assertIsNone(row['right_arm_elevation_deg'])
        self.assertIsNone(row['elevation_diff_signed_deg'])
        self.assertEqual(row['status'], 'partial_tracking')

    def test_low_presence_is_preserved_without_changing_visibility_rule(self):
        points = pose(90, 45)
        points[nf.JOINTS['left']['elbow']].presence = .391
        row = measure_bilateral_frame(0, 5, points, self.settings, 1000, 1000)
        self.assertAlmostEqual(row['left_elbow_presence'], .391)
        self.assertEqual((row['left_usable'], row['right_usable']), (1, 1))
        self.assertAlmostEqual(row['left_arm_elevation_deg'], 90)
        self.assertAlmostEqual(row['right_arm_elevation_deg'], 45)


class TestBilateralDetector(unittest.TestCase):
    def test_rolling_stable_rest_recovers_after_slightly_unreliable_first_frame(self):
        samples = [(0, None), (50, 3.57), (90, 5.59), (100, 8.4),
                   (200, 7.2), (300, 6.6), (400, 7.3), (500, 7.5),
                   (570, 10.55), (610, 8.9)]
        current = BilateralRepetitionDetector(25)
        historical = BilateralRepetitionDetector(25, reliable_baseline=False)
        for t, left in samples:
            self.assertIsNone(current.push(t, left, 5))
            historical.push(t, left, 5)
        self.assertIsNotNone(current.arms['left'].baseline)
        self.assertIsNone(historical.arms['left'].baseline)
        self.assertGreaterEqual(current.arms['left'].rest[-1][0] - current.arms['left'].rest[0][0], 500)
        self.assertLessEqual(max(v for _, v in current.arms['left'].rest) -
                             min(v for _, v in current.arms['left'].rest), 5)
        self.assertEqual(current.instruction, 'BOTH READY - raise both arms once')

    def test_movement_before_both_baselines_is_explicitly_unjudgeable(self):
        detector = BilateralRepetitionDetector(25)
        result = None
        for t in range(0, 1000, 100):
            left = None if t == 0 else (5 if t < 500 else 30 + (t - 500) / 10)
            right = 5 if t < 600 else 90
            result = detector.push(t, left, right)
            if result: break
        self.assertIsNotNone(result)
        self.assertEqual(result.outcome, 'unable_to_judge')
        self.assertEqual(result.termination_reason, 'movement_before_both_ready')
        self.assertIsNone(result.arm_results['left'].rest_baseline_deg)

    def test_false_preonset_peak_is_excluded_from_movement_interval(self):
        detector = BilateralRepetitionDetector(25)
        result = None
        for t in range(0, 3000, 100):
            left = (139.58 if t == 0 else 16.2 if t == 100 else
                    5 if t < 800 or t >= 2300 else 90)
            right = 5 if t < 800 or t >= 2300 else 88
            result = detector.push(t, left, right)
            if result: break
        self.assertEqual(result.outcome, 'complete')
        self.assertEqual(result.peaks_deg['left'], 90)
        self.assertGreaterEqual(result.peak_times_ms['left'], result.arm_results['left'].start_ms)
        self.assertLessEqual(result.peak_times_ms['left'], result.arm_results['left'].end_ms)

    def test_simultaneous_and_staggered_complete_returns(self):
        for starts, returns in (((600, 600), (2000, 2000)),
                                ((600, 900), (2000, 2500)),
                                ((900, 600), (2500, 2000))):
            detector, result, _ = cycle(*starts, *returns)
            self.assertEqual(result.outcome, 'complete')
            self.assertEqual(result.termination_reason, 'both_confirmed_return')
            self.assertEqual(result.start_ms, min(starts))
            self.assertEqual(result.end_ms, max(returns))
            self.assertEqual(result.metadata()['arm_start_ms'], dict(zip(('left', 'right'), starts)))
            self.assertEqual(result.metadata()['arm_end_ms'], dict(zip(('left', 'right'), returns)))

    def test_one_arm_return_does_not_complete_bilateral(self):
        detector, result, _ = cycle(left_return=2000, right_return=None, until=26000)
        self.assertEqual(result.outcome, 'incomplete_return')
        self.assertEqual(result.termination_reason, 'movement_timeout')
        self.assertEqual(result.arm_results['left'].outcome, 'complete')
        self.assertNotEqual(result.arm_results['right'].outcome, 'complete')

    def test_one_frame_loss_recovers_but_sustained_loss_abstains(self):
        _, recovered, _ = cycle(missing=(1000, 1000, 'right'))
        self.assertEqual(recovered.outcome, 'complete')
        _, lost, _ = cycle(missing=(1000, 1400, 'right'))
        self.assertEqual(lost.outcome, 'unable_to_judge')
        self.assertEqual(lost.termination_reason, 'right_tracking_lost')
        _, both, _ = cycle(missing=(1000, 1400, 'both'))
        self.assertEqual(both.outcome, 'unable_to_judge')

    def test_lagging_peak_and_low_elevation_still_complete(self):
        detector = BilateralRepetitionDetector(25)
        result = None
        for t in range(0, 3100, 100):
            left = 5 if t < 600 or t >= 2000 else (95 if t >= 1000 else 40)
            right = 5 if t < 600 or t >= 2400 else (50 if t >= 1500 else 30)
            result = detector.push(t, left, right)
            if result: break
        self.assertEqual(result.outcome, 'complete')
        self.assertEqual(result.peaks_deg, {'left': 95, 'right': 50})
        self.assertLess(result.peak_times_ms['left'], result.peak_times_ms['right'])

    def test_no_start_and_invalid_timestamp(self):
        _, result, _ = cycle(left_start=9000, right_start=9000, until=9000)
        self.assertEqual(result.outcome, 'no_start')
        detector = BilateralRepetitionDetector(25)
        detector.push(0, 5, 5)
        self.assertEqual(detector.push(0, 5, 5).termination_reason, 'invalid_timestamp')

    def test_initial_254_or_270_ms_gap_restarts_rest_without_changing_time(self):
        for gap in (254, 270):
            detector = BilateralRepetitionDetector(25)
            self.assertIsNone(detector.push(0, 5, 5))
            self.assertIsNone(detector.push(gap, 5, 5))
            self.assertEqual(detector.last_ms, gap)
            self.assertEqual(detector.first_ms, gap)
            self.assertEqual(detector.startup_gap_restarts, 1)
            self.assertEqual(detector.state, 'left: rest_calibrating | right: rest_calibrating')
            self.assertEqual(detector.push(gap + 270, 5, 5).termination_reason,
                             'startup_frame_gap')
        historical = BilateralRepetitionDetector(25, restart_startup_gap=False)
        historical.push(0, 5, 5)
        self.assertEqual(historical.push(270, 5, 5).termination_reason, 'frame_gap')

    def test_active_movement_gap_still_abstains(self):
        detector = BilateralRepetitionDetector(25)
        for t in range(0, 1000, 100):
            detector.push(t, 5 if t < 600 else 90, 5 if t < 600 else 90)
        self.assertEqual(detector.arms['left'].state, 'moving')
        result = detector.push(1254, 90, 90)
        self.assertEqual(result.outcome, 'unable_to_judge')
        self.assertEqual(result.termination_reason, 'frame_gap')
        self.assertEqual(detector.startup_gap_restarts, 0)


class TestBilateralIsolationAndCapture(unittest.TestCase):
    def test_historical_schema_four_and_five_replay_with_original_peak_policy(self):
        with TemporaryDirectory() as folder:
            settings = {**load_settings(Path(folder) / 'settings.json'), 'arm': 'both'}
            for schema in (4, 5):
                detector = BilateralRepetitionDetector(
                    25, restart_startup_gap=schema >= 5, reliable_baseline=False)
                rows = []
                for t in range(0, 3000, 100):
                    left = 139.58 if t == 0 else (5 if t < 800 or t >= 2000 else 90)
                    right = 5 if t < 800 or t >= 2000 else 88
                    result = detector.push(t, left, right)
                    row = {'frame': len(rows), 'timestamp_ms': t, 'arm': 'both',
                           'angle_definition': settings['angle_definition'],
                           'left_arm_elevation_deg': left, 'right_arm_elevation_deg': right,
                           'left_usable': 1, 'right_usable': 1,
                           'elevation_diff_signed_deg': left - right,
                           'elevation_diff_abs_deg': abs(left - right),
                           'landmarks': [{}] * 33, 'detector_state': detector.state}
                    if schema == 5:
                        row['startup_gap_reset'] = detector.startup_gap_reset
                    rows.append(row)
                    if result: break
                path = Path(folder) / f'schema_{schema}.avi'
                path.with_suffix('.timestamps.csv').write_text(
                    'frame,timestamp_ms\n' + ''.join(f'{row["frame"]},{row["timestamp_ms"]}\n' for row in rows),
                    encoding='utf-8')
                path.with_suffix('.live_pose.jsonl').write_text(
                    ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                metadata = {**result.metadata(), 'recording_schema_version': schema,
                            'arm': 'both', 'min_visibility': settings['min_visibility'],
                            'angle_definition': settings['angle_definition'],
                            'capture_frames': len(rows), 'termination_elapsed_ms': rows[-1]['timestamp_ms'],
                            'rep_timeout_s': 25}
                if schema == 5:
                    metadata.update(startup_gap_policy='one_rest_baseline_restart',
                                    startup_gap_restarts=0)
                replay = load_bilateral_live_pose(path, settings, metadata)
                self.assertEqual(replay['repetition'].outcome, 'complete')
                self.assertEqual(replay['repetition'].peaks_deg['left'], 139.58)

    def test_first_then_second_diagnostic_attempt_through_menu(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            settings = {**load_settings(root / 'settings.json'), 'arm': 'both'}
            choices = iter(['17', '', '17', '', '0'])
            model_creations = []
            calls = []
            prompts = []

            def launch(args, cwd, check):
                attempt = len(calls)
                calls.append(args[-1])
                self.assertEqual(args[-2], 'diagnostic')
                camera = SimpleNamespace(read=lambda: (True, np.zeros((32, 32, 3), dtype=np.uint8)),
                                         get=lambda key: 10.0, release=lambda: None)
                capture_times = ([5.20, 5.47] + [5.57 + i / 10 for i in range(50)]
                                 if attempt == 0 else [5 + i / 10 for i in range(60)])
                setup_ready = 5.20 if attempt == 0 else 5.0
                clock = iter([0.0, 0.0, 0.0, 5.0, setup_ready] + capture_times)
                keys = iter([ord(' ')] + [255] * 100)

                def model():
                    model_creations.append(attempt)
                    return nullcontext(object())

                def detect(detector, frame, model_timestamp):
                    elapsed = max(0, model_timestamp - 5000)
                    angle = 5 if elapsed < 1000 or elapsed >= 2500 else 90
                    return pose(angle, angle)

                with (patch.object(sys, 'argv', args[1:]),
                      patch('step4_record.nf.open_camera', return_value=(camera, 32, 32)),
                      patch('step4_record.nf.make_landmarker', side_effect=model),
                      patch('step4_record.nf.detect', side_effect=detect),
                      patch('step4_record.overlay', side_effect=lambda frame, label, *args: prompts.append(label)),
                      patch('step4_record.time.monotonic', side_effect=lambda: next(clock)),
                      patch.multiple('cv2', namedWindow=lambda *a: None, resizeWindow=lambda *a: None,
                                     imshow=lambda *a: None, waitKey=lambda delay: next(keys),
                                     destroyAllWindows=lambda: None)):
                    step4_record.main()

            with (patch('builtins.input', side_effect=lambda prompt: next(choices)),
                  patch('menu.nf.load_settings', return_value=settings),
                  patch('menu.nf.mode_dir', side_effect=lambda kind, active: root / mode_dir(kind, active)),
                  patch('menu.subprocess.run', side_effect=launch),
                  redirect_stdout(io.StringIO())):
                menu.main()
            self.assertEqual(calls, ['D_01', 'D_02'])
            self.assertEqual(model_creations, [0, 1])
            self.assertIn('WAIT: BOTH ARMS DOWN', prompts)
            self.assertIn('BOTH READY - RAISE', prompts)
            for name, expected_gap, expected_reset in (('D_01', 270, 1), ('D_02', 100, 0)):
                path = root / mode_dir('clips', settings) / 'diagnostic' / f'{name}.avi'
                metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                replay = load_bilateral_live_pose(path, settings, metadata)
                self.assertEqual(metadata['capture_outcome'], 'complete')
                self.assertEqual(metadata['startup_gap_restarts'], expected_reset)
                self.assertEqual(replay['rows'][1]['timestamp_ms'], expected_gap)
                self.assertEqual(replay['rows'][1]['startup_gap_reset'], bool(expected_reset))
                self.assertEqual([row['frame'] for row in replay['rows']],
                                 list(range(len(replay['rows']))))
                video = cv2.VideoCapture(str(path))
                try:
                    self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), len(replay['rows']))
                finally:
                    video.release()
    def test_arm_settings_and_paths_are_separate(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            right = load_settings(path)
            right.update(reference_clip='right.avi', dtw_threshold=2.224,
                         dtw_calibration={'id': 'right'})
            save_settings(right, path)
            left = set_arm('left', path)
            left.update(reference_clip='left.avi', dtw_threshold=4.056,
                        dtw_calibration={'id': 'left'})
            save_settings(left, path)
            both = set_arm('both', path)
            self.assertIsNone(both['dtw_threshold'])
            self.assertIsNone(both['dtw_calibration'])
            self.assertIsNone(both['reference_clip'])
            self.assertEqual(mode_dir('clips', both), Path('clips/seated/both/segmented_v1'))
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                save_settings({**both, 'dtw_threshold': 2.224}, path)
            self.assertEqual(set_arm('right', path)['dtw_threshold'], 2.224)
            self.assertEqual(set_arm('left', path)['dtw_threshold'], 4.056)
            set_arm('both', path)
            with self.assertRaisesRegex(ValueError, 'seated only'):
                set_position('standing', path)

    def test_menu_blocks_bilateral_scoring_and_freezing(self):
        with TemporaryDirectory() as folder:
            settings = {**load_settings(Path(folder) / 'settings.json'), 'arm': 'both'}
            with (patch('builtins.input', side_effect=['8', '9', '11', '12', '0']),
                  patch('menu.nf.load_settings', return_value=settings),
                  patch('menu.run') as launch,
                  redirect_stdout(io.StringIO())):
                menu.main()
            launch.assert_not_called()

    def test_bilateral_test_capture_does_not_inherit_unilateral_calibration(self):
        with TemporaryDirectory() as folder:
            settings = {**load_settings(Path(folder) / 'settings.json'), 'arm': 'both'}
            with (patch.object(sys, 'argv', ['step4_record.py', 'test', 'T_01']),
                  patch('step4_record.nf.load_settings', return_value=settings),
                  patch('step4_record.nf.mode_dir', side_effect=lambda kind, active: Path(folder) / mode_dir(kind, active)),
                  patch('step4_record.require_calibration') as calibration,
                  patch('step4_record.nf.open_camera', side_effect=RuntimeError('camera marker')),
                  self.assertRaisesRegex(RuntimeError, 'camera marker')):
                step4_record.main()
            calibration.assert_not_called()

    def test_menu_recorder_writes_synchronized_pose_and_replays_it(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            settings = {**load_settings(root / 'settings.json'), 'arm': 'both'}
            camera = SimpleNamespace(read=lambda: (True, np.zeros((32, 32, 3), dtype=np.uint8)),
                                     get=lambda key: 10.0, release=lambda: None)
            clock = iter([0.0, 0.0, 0.0, 5.0, 5.0] + [5 + tick / 10 for tick in range(80)])
            keys = iter([ord(' ')] + [255] * 80)
            choices = iter(['17', '', '0'])
            detections = []

            def detect(detector, frame, timestamp):
                t = max(0, timestamp - 5000)
                left = 5 if t < 600 or t >= 2000 else 90
                right = 5 if t < 900 or t >= 2400 else 50
                points = pose(left, right)
                if t == 1200:
                    points[nf.JOINTS['right']['elbow']].visibility = 0
                detections.append(t)
                return points

            def launch(args, cwd, check):
                with patch.object(sys, 'argv', args[1:]):
                    step4_record.main()

            with (patch('builtins.input', side_effect=lambda prompt: next(choices)),
                  patch('menu.subprocess.run', side_effect=launch),
                  patch('menu.nf.load_settings', return_value=settings),
                  patch('menu.nf.mode_dir', side_effect=lambda kind, active: root / mode_dir(kind, active)),
                  patch('step4_record.nf.open_camera', return_value=(camera, 32, 32)),
                  patch('step4_record.nf.make_landmarker', side_effect=lambda: nullcontext(object())),
                  patch('step4_record.nf.detect', side_effect=detect),
                  patch('step4_record.time.monotonic', side_effect=lambda: next(clock)),
                  patch.multiple('cv2', namedWindow=lambda *args: None, resizeWindow=lambda *args: None,
                                 imshow=lambda *args: None, waitKey=lambda delay: next(keys),
                                 destroyAllWindows=lambda: None),
                  redirect_stdout(io.StringIO())):
                menu.main()
            path = root / mode_dir('clips', settings) / 'diagnostic' / 'D_01.avi'
            metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            with path.with_suffix('.timestamps.csv').open(newline='', encoding='utf-8') as source:
                times = [int(row['timestamp_ms']) for row in csv.DictReader(source)]
            result = load_bilateral_live_pose(path, settings, metadata)
            rows = result['rows']
            self.assertEqual(metadata['recording_schema_version'], 6)
            self.assertEqual(metadata['arm'], 'both')
            self.assertEqual(metadata['capture_outcome'], 'complete')
            self.assertEqual(len(rows), len(times))
            self.assertEqual([row['frame'] for row in rows], list(range(len(rows))))
            self.assertEqual([row['timestamp_ms'] for row in rows], times)
            self.assertEqual(len(detections), len(rows) + 1)  # countdown warm-up is not saved
            self.assertEqual(metadata['arm_start_ms'], {'left': 600, 'right': 900})
            self.assertEqual(metadata['arm_end_ms'], {'left': 2000, 'right': 2400})
            video = cv2.VideoCapture(str(path))
            try:
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), len(rows))
            finally:
                video.release()
            self.assertEqual(result['repetition'].metadata()['arm_outcomes'],
                             {'left': 'complete', 'right': 'complete'})
            self.assertIsNone(rows[0]['arm_elevation_deg'])
            self.assertEqual(rows[0]['left_usable'], 1)
            self.assertEqual(rows[0]['right_usable'], 1)
            self.assertEqual(rows[12]['timestamp_ms'], 1200)
            self.assertEqual((rows[12]['left_usable'], rows[12]['right_usable']), (1, 0))
            self.assertIsNone(rows[12]['right_arm_elevation_deg'])
            self.assertIsNone(rows[12]['elevation_diff_signed_deg'])
            stamp_file = path.with_suffix('.timestamps.csv')
            original = stamp_file.read_text(encoding='utf-8')
            lines = original.splitlines()
            lines[1] = lines[1].replace('0,', '1,', 1)
            stamp_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'noncontiguous'):
                load_bilateral_live_pose(path, settings, metadata)


if __name__ == '__main__':
    unittest.main()
