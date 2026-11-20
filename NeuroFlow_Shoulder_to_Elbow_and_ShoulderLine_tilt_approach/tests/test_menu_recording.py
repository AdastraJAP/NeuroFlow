import csv
import io
import json
import shutil
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
import step4_record
import step5_score
from config import load_settings, mode_dir
from processing import live_pose_path
from segmented_scoring import detect_clip, reference_identity, scoring_dir, scoring_measurements


class SimulatedCamera:
    def __init__(self):
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.released = False

    def read(self):
        return True, self.frame

    def get(self, property_id):
        return 10.0

    def release(self):
        self.released = True


class TestMenuRecording(unittest.TestCase):
    def run_option(self, option, category, name, return_at_ms):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / 'ref.avi'
            reference.write_bytes(b'test reference identity')
            reference.with_suffix('.timestamps.csv').write_text('frame,timestamp_ms\n0,0\n')
            reference.with_suffix('.json').write_text('{}')
            settings = load_settings(root / 'settings.json')
            settings.update(reference_clip=str(reference), dtw_threshold=2.0)
            settings['dtw_calibration'] = {
                'id': 'synthetic-calibration', 'reference': reference_identity(settings), 'threshold': 2.0,
            }
            camera = SimulatedCamera()
            clock = iter([0.0, 0.0] + [5 + tick / 10 for tick in range(400)])
            keys = iter([ord(' ')] + [255] * 400)
            choices = iter([option, *(['incomplete'] if option == '10' else []), '', '0'])

            def pose_angle(landmarks, *args):
                stamp = landmarks[0].z
                at_rest = stamp < 600 or (return_at_ms is not None and stamp >= return_at_ms)
                return (5.12345678901234 if at_rest else 90.12345678901234), True

            def launch(args, cwd, check):
                self.assertEqual(args, [sys.executable, str(menu.PROJECT_DIR / 'step4_record.py'), category, name])
                self.assertEqual(cwd, menu.PROJECT_DIR)
                with patch.object(sys, 'argv', args[1:]):
                    step4_record.main()

            with (
                patch('builtins.input', side_effect=lambda prompt: next(choices)),
                patch('menu.subprocess.run', side_effect=launch) as launcher,
                patch('menu.nf.load_settings', return_value=settings),
                patch('menu.nf.mode_dir', side_effect=lambda kind, active: root / mode_dir(kind, active)),
                patch('step4_record.nf.open_camera', return_value=(camera, 32, 32)),
                patch('step4_record.nf.make_landmarker', side_effect=lambda: nullcontext(object())),
                patch('step4_record.nf.detect', side_effect=lambda detector, frame, timestamp: [
                    SimpleNamespace(x=0.234567890123456, y=0.5, z=timestamp, visibility=0.99, presence=1.0)
                ] * 33),
                patch('step4_record.nf.shoulder_tilt', return_value=None),
                patch('step4_record.nf.shoulder_angle', side_effect=pose_angle),
                patch('step4_record.time.monotonic', side_effect=lambda: next(clock)),
                patch.multiple('cv2', namedWindow=lambda *args: None, resizeWindow=lambda *args: None,
                               imshow=lambda *args: None, waitKey=lambda delay: next(keys),
                               destroyAllWindows=lambda: None),
                redirect_stdout(io.StringIO()),
            ):
                menu.main()
            launcher.assert_called_once()
            self.assertTrue(camera.released)
            path = root / mode_dir('clips', settings) / category / name
            metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            with path.with_suffix('.timestamps.csv').open(newline='', encoding='utf-8') as source:
                timestamps = [int(row['timestamp_ms']) for row in csv.DictReader(source)]
            video = cv2.VideoCapture(str(path.with_suffix('.avi')))
            try:
                self.assertTrue(video.isOpened())
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), len(timestamps))
            finally:
                video.release()
            self.assertEqual(metadata['version'], settings['version'])
            self.assertNotIn('recording_duration_requested_s', metadata)
            self.assertEqual(metadata['rep_timeout_s'], 25)
            self.assertEqual(metadata['termination_elapsed_ms'], timestamps[-1])
            self.assertEqual(metadata['repetition_start_ms'], 600)
            replay, repetition = scoring_measurements(path.with_suffix('.avi'), settings, metadata)
            self.assertEqual(repetition.outcome, metadata['detector_status'])
            self.assertEqual(repetition.end_ms, metadata['repetition_end_ms'])
            self.assertEqual(repetition.rest_baseline_deg, 5.12345678901234)
            self.assertEqual(replay['rows'][0]['landmarks'][0]['x'], 0.234567890123456)
            self.assertEqual(replay['rows'][6]['arm_elevation_deg'], 90.12345678901234)
            with self.assertRaisesRegex(ValueError, 'replay disagrees'):
                scoring_measurements(path.with_suffix('.avi'), settings,
                                     {**metadata, 'repetition_start_ms': 601})
            if option == '5':
                delayed_return = [
                    {**row, 'arm_elevation_deg': row['arm_elevation_deg'] + 8}
                    if return_at_ms <= row['timestamp_ms'] < return_at_ms + 200 else row
                    for row in replay['rows']
                ]
                self.assertEqual(detect_clip({'rows': delayed_return}, settings).outcome, 'incomplete_return')
                self.score_saved_practice(root, settings, reference, path, metadata)
            if option == '10':
                self.assertEqual(metadata['calibration_id'], 'synthetic-calibration')
                self.assertEqual((path.parent / 'expected_labels.csv').read_text().splitlines(),
                                 ['clip,expected_label', f'{name},incomplete'])
            return metadata

    def score_saved_practice(self, root, settings, reference, path, metadata):
        for suffix in ('.avi', '.timestamps.csv', '.live_pose.jsonl'):
            shutil.copyfile(path.with_suffix(suffix), reference.with_suffix(suffix))
        reference.with_suffix('.json').write_text(json.dumps({**metadata, 'category': 'reference'}))
        legacy = path.parent / 'complete_legacy.avi'
        shutil.copyfile(path.with_suffix('.avi'), legacy)
        legacy.with_suffix('.json').write_text(json.dumps({**metadata, 'recording_schema_version': 2}))
        historical = root / mode_dir('traces', settings) / 'summary_practice.csv'
        historical.parent.mkdir(parents=True)
        historical.write_text('previous evaluation')
        with (
            patch.object(sys, 'argv', ['step5_score.py']),
            patch('menu.nf.load_settings', return_value={**settings}),
            patch('menu.nf.mode_dir', side_effect=lambda kind, active: root / mode_dir(kind, active)),
            patch('segmented_scoring.process_clip', side_effect=AssertionError('Unexpected video replay')),
            redirect_stdout(io.StringIO()),
        ):
            step5_score.main()
        self.assertEqual(historical.read_text(), 'previous evaluation')
        summary = historical.parent / scoring_dir(settings).name / 'summary_practice.csv'
        with summary.open(newline='', encoding='utf-8') as source:
            scored, old = list(csv.DictReader(source))
        self.assertEqual(scored['measurement_source'], 'live_capture')
        self.assertEqual(scored['repetition_outcome'], 'complete')
        self.assertEqual(int(scored['end_ms']), metadata['repetition_end_ms'])
        self.assertEqual(float(scored['dtw_score']), 0.0)
        self.assertEqual(old['termination_reason'], 'legacy_pose_unavailable')
        self.assertEqual(old['dtw_score'], '')
        live_pose_path(path).unlink()
        with self.assertRaises(FileNotFoundError):
            scoring_measurements(path.with_suffix('.avi'), settings, metadata)

    def test_option_5_records_past_ten_seconds_and_stops_after_return_dwell(self):
        metadata = self.run_option('5', 'practice', 'complete_01', 13000)
        self.assertEqual(metadata['detector_status'], 'complete')
        self.assertEqual(metadata['termination_reason'], 'confirmed_return')
        self.assertEqual(metadata['repetition_end_ms'], 13000)
        self.assertEqual(metadata['repetition_duration_s'], 12.4)
        self.assertEqual(metadata['actual_duration_s'], 13.4)

    def test_option_6_stops_early_when_return_is_confirmed(self):
        metadata = self.run_option('6', 'practice', 'incomplete_01', 2000)
        self.assertEqual(metadata['termination_reason'], 'confirmed_return')
        self.assertEqual(metadata['actual_duration_s'], 2.4)

    def test_option_10_stops_at_25_seconds_from_movement_onset(self):
        metadata = self.run_option('10', 'test', 'T_01', None)
        self.assertEqual(metadata['detector_status'], 'incomplete_return')
        self.assertEqual(metadata['termination_reason'], 'movement_timeout')
        self.assertEqual(metadata['actual_duration_s'], 25.6)
        self.assertIsNone(metadata['repetition_end_ms'])
        self.assertIsNone(metadata['repetition_duration_s'])
