"""A5 arm selection, geometry, configuration, and legacy replay regression tests."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import menu
import nf_core as nf
import step5_score
from config import load_settings, mode_dir, save_settings, set_arm, set_position
from processing import load_live_pose, measure_frame
from repetition import Repetition, detect_repetition
from segmented_scoring import compare, require_calibration, scoring_dir


def landmarks():
    points = [SimpleNamespace(x=0.5, y=1.5, z=0.0, visibility=0.0, presence=0.0)
              for _ in range(33)]
    for arm, x in (('left', 0.3), ('right', 0.7)):
        points[nf.JOINTS[arm]['shoulder']] = SimpleNamespace(
            x=x, y=0.3, z=0.0, visibility=0.99, presence=0.99)
        points[nf.JOINTS[arm]['elbow']] = SimpleNamespace(
            x=x, y=0.6, z=0.0, visibility=0.99, presence=0.99)
    return points


class TestLeftArmGeometry(unittest.TestCase):
    def test_selected_arm_zero_45_90_without_hip_or_wrist(self):
        points = landmarks()
        left_elbow = points[nf.JOINTS['left']['elbow']]
        for x, y, expected in ((0.3, 0.6, 0), (0.15, 0.45, 45), (0.15, 0.3, 90)):
            left_elbow.x, left_elbow.y = x, y
            angle, usable = nf.shoulder_angle(points, 'left', 1000, 1000, 0.5)
            self.assertTrue(usable)
            self.assertAlmostEqual(angle, expected)
        self.assertAlmostEqual(nf.shoulder_angle(points, 'right', 1000, 1000, 0.5)[0], 0)
        self.assertEqual(points[nf.JOINTS['left']['hip']].visibility, 0)
        self.assertEqual(points[nf.JOINTS['left']['wrist']].visibility, 0)

    def test_horizontal_preview_mirror_does_not_change_angle(self):
        points = landmarks()
        points[nf.JOINTS['left']['elbow']].x = 0.15
        points[nf.JOINTS['left']['elbow']].y = 0.45
        original = nf.shoulder_angle(points, 'left', 1000, 1000, 0.5)[0]
        for point in points:
            point.x = 1.0 - point.x
        self.assertAlmostEqual(nf.shoulder_angle(points, 'left', 1000, 1000, 0.5)[0], original)

    def test_preview_flip_keeps_anatomical_labels(self):
        points = landmarks()
        left = points[nf.JOINTS['left']['shoulder']]
        right = points[nf.JOINTS['right']['shoulder']]
        self.assertEqual(nf._screen_pt(left, 1000, 1000, mirror=True)[0], 700)
        self.assertEqual(nf._screen_pt(right, 1000, 1000, mirror=True)[0], 300)
        self.assertAlmostEqual(nf.shoulder_angle(points, 'left', 1000, 1000, 0.5)[0], 0)

    def test_right_arm_movement_does_not_trigger_selected_left_arm(self):
        points = landmarks()
        points[nf.JOINTS['right']['elbow']].x = 0.55
        points[nf.JOINTS['right']['elbow']].y = 0.3
        settings = {'arm': 'left', 'min_visibility': 0.5}
        row = measure_frame(0, 0, points, settings, 1000, 1000)
        self.assertEqual(row['arm'], 'left')
        self.assertAlmostEqual(row['arm_elevation_deg'], 0)
        self.assertAlmostEqual(nf.shoulder_angle(points, 'right', 1000, 1000, 0.5)[0], 90)
        outcome = detect_repetition([(t, row['arm_elevation_deg']) for t in range(0, 8200, 100)], 25)
        self.assertEqual(outcome.outcome, 'no_start')

    def test_completed_low_elevation_is_not_exercise_success(self):
        repetition = Repetition('complete', 600, 2000,
                                ((600, 5.0), (1000, 50.0), (2000, 5.0)),
                                'confirmed_return', 5.0)
        settings = {'movement_target_deg': 90.0, 'target_tolerance_deg': 10.0,
                    'dtw_threshold': None}
        peak, _, message, _ = compare(repetition, [5.0] * 101, settings)
        self.assertEqual(peak, 50.0)
        self.assertEqual(message, 'Arm elevation below target')


class TestLeftArmConfiguration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings_path = Path(self.temp.name) / 'settings.json'

    def test_arm_and_position_calibrations_and_paths_are_isolated(self):
        right = load_settings(self.settings_path)
        right['reference_clip'] = 'clips/seated/segmented_v1/reference/ref_01.avi'
        right['dtw_threshold'] = 2.224
        right['dtw_calibration'] = {'id': 'right-only'}
        save_settings(right, self.settings_path)
        left = set_arm('left', self.settings_path)
        self.assertIsNone(left['reference_clip'])
        self.assertIsNone(left['dtw_threshold'])
        self.assertIsNone(left['dtw_calibration'])
        with self.assertRaisesRegex(SystemExit, 'fresh practice calibration'):
            require_calibration(left)
        self.assertEqual(mode_dir('clips', left), Path('clips/seated/left/segmented_v1'))
        self.assertEqual(mode_dir('traces', right), Path('traces/seated/segmented_v1'))
        self.assertNotEqual(scoring_dir(left), scoring_dir(right))
        left['reference_clip'] = 'clips/seated/left/segmented_v1/reference/ref_01.avi'
        left['dtw_threshold'] = 4.5
        left['dtw_calibration'] = {'id': 'left-only'}
        save_settings(left, self.settings_path)
        standing_left = set_position('standing', self.settings_path)
        self.assertIsNone(standing_left['reference_clip'])
        self.assertIsNone(standing_left['dtw_threshold'])
        self.assertEqual(set_position('seated', self.settings_path)['dtw_threshold'], 4.5)
        right_again = set_arm('right', self.settings_path)
        self.assertEqual(right_again['dtw_threshold'], 2.224)
        self.assertEqual(right_again['dtw_calibration']['id'], 'right-only')
        self.assertEqual(mode_dir('clips', right_again), Path('clips/seated/segmented_v1'))
        raw = json.loads(self.settings_path.read_text(encoding='utf-8'))
        self.assertEqual(raw['conditions']['seated']['dtw_threshold'], 2.224)
        self.assertEqual(raw['arm_conditions']['left']['seated']['dtw_threshold'], 4.5)

    def test_real_right_threshold_is_read_only_and_left_starts_uncalibrated(self):
        source = Path(__file__).resolve().parents[1] / 'settings.json'
        before = source.read_bytes()
        stored = json.loads(before)
        self.assertEqual(stored['conditions']['seated']['dtw_threshold'], 2.224)
        self.assertIsNone(set_arm('left', self.settings_path)['dtw_threshold'])
        self.assertEqual(source.read_bytes(), before)

    def test_menu_switches_anatomical_arm(self):
        settings = load_settings(self.settings_path)
        with (patch('builtins.input', side_effect=['16', '', '0']),
              patch('menu.nf.load_settings', return_value=settings),
              patch('menu.nf.set_arm', return_value={**settings, 'arm': 'left'}) as choose,
              redirect_stdout(io.StringIO())):
            menu.main()
        choose.assert_called_once_with('left')

    def test_freezing_left_practice_threshold_preserves_right_2224(self):
        right = load_settings(self.settings_path)
        right['reference_clip'] = 'right-reference.avi'
        right['dtw_threshold'] = 2.224
        save_settings(right, self.settings_path)
        left = set_arm('left', self.settings_path)
        left['reference_clip'] = 'left-reference.avi'
        save_settings(left, self.settings_path)
        traces = Path(self.temp.name) / 'left-practice-traces'
        traces.mkdir()
        identity = {'arm': 'left', 'reference_clip': 'left-reference.avi'}
        (traces / 'practice_meta.json').write_text(json.dumps(identity), encoding='utf-8')
        with (traces / 'summary_practice.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['clip', 'dtw_score', 'max_arm_elevation_deg'])
            writer.writeheader()
            for i, score in enumerate((1.0, 1.2, 1.4), 1):
                writer.writerow({'clip': f'complete_{i:02d}', 'dtw_score': score,
                                 'max_arm_elevation_deg': 90})
            for i, score in enumerate((4.0, 4.5, 5.0), 1):
                writer.writerow({'clip': f'incomplete_{i:02d}', 'dtw_score': score,
                                 'max_arm_elevation_deg': 50})
        with (patch('step5_score.nf.save_settings', side_effect=lambda s: save_settings(s, self.settings_path)),
              redirect_stdout(io.StringIO())):
            step5_score.freeze('2.5', left, traces, identity)
        self.assertEqual(load_settings(self.settings_path)['dtw_threshold'], 2.5)
        self.assertEqual(set_arm('right', self.settings_path)['dtw_threshold'], 2.224)

    def test_historical_schema_three_right_sidecar_without_arm_replays(self):
        video = Path(self.temp.name) / 'old.avi'
        video.with_suffix('.timestamps.csv').write_text('frame,timestamp_ms\n0,0\n', encoding='utf-8')
        row = {'frame': 0, 'timestamp_ms': 0, 'pose_detected': 1,
               'arm_elevation_deg': 10.0, 'shoulder_tilt_deg': 0.0,
               'usable': 1, 'status': 'ok', 'landmarks': [{}] * 33}
        video.with_suffix('.live_pose.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
        settings = load_settings(self.settings_path)
        metadata = {'recording_schema_version': 3, 'capture_frames': 1,
                    'termination_elapsed_ms': 0, 'min_visibility': settings['min_visibility'],
                    'arm': 'right'}
        result = load_live_pose(video, settings, metadata)
        self.assertEqual(result['rows'][0]['arm_elevation_deg'], 10.0)
        with self.assertRaisesRegex(ValueError, 'recorded arm differs'):
            load_live_pose(video, {**settings, 'arm': 'left'}, metadata)


if __name__ == '__main__':
    unittest.main()
