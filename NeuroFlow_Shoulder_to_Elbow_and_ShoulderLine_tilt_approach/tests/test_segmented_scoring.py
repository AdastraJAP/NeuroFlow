import unittest
from unittest.mock import patch

from config import DEFAULT_SETTINGS
from repetition import detect_repetition, resample
from segmented_scoring import compare, detect_clip, require_calibration
from step5_score import practice_bounds
from step6_demo import score_rep


class TestSegmentedScoring(unittest.TestCase):
    def setUp(self):
        self.settings = {**DEFAULT_SETTINGS, 'dtw_threshold': 2.0,
                         'dtw_calibration': {'id': 'synthetic-test'}}
        angles = [5] * 6 + [28, 60, 90, 60, 29] + [6] * 5
        self.samples = [(index * 100, angle, True) for index, angle in enumerate(angles)]
        self.repetition = detect_repetition([(timestamp, angle) for timestamp, angle, _ in self.samples], 25)
        self.reference = resample(self.repetition.segment)

    def test_live_and_offline_score_same_segment(self):
        offline = detect_clip({'rows': [
            {'timestamp_ms': timestamp, 'arm_elevation_deg': angle, 'usable': usable}
            for timestamp, angle, usable in self.samples
        ]}, self.settings)
        peak, score, message, _ = compare(offline, self.reference, self.settings)
        live, live_message, _, live_score = score_rep(
            self.repetition, self.samples, [], None, 'complete', self.settings, self.reference)
        self.assertEqual((peak, score, message), (90, 0, 'Target reached'))
        self.assertEqual((live['start_ms'], live['end_ms']), (600, 1100))
        self.assertEqual((live_message, live_score), (message, score))

    def test_incomplete_and_missing_have_no_dtw_score(self):
        for ending, expected in [([], 'Incomplete return'), ([(900, None)], 'Unable to judge')]:
            with self.subTest(expected=expected):
                samples = [(timestamp, angle) for timestamp, angle, _ in self.samples[:9]]
                repetition = detect_repetition(samples + ending, 25)
                _, score, message, _ = compare(repetition, self.reference, self.settings)
                self.assertIsNone(score)
                self.assertTrue(message.startswith(expected))

    def test_calibration_cannot_use_legacy_or_edited_threshold(self):
        self.settings['reference_clip'] = 'reference.avi'
        with self.assertRaises(SystemExit):
            require_calibration({**self.settings, 'dtw_threshold': 1.48, 'dtw_calibration': None})
        self.settings['dtw_calibration'].update(reference={'position': 'seated'}, threshold=2.0)
        with patch('segmented_scoring.reference_identity', return_value={'position': 'seated'}):
            require_calibration(self.settings)
            with self.assertRaises(SystemExit):
                require_calibration({**self.settings, 'dtw_threshold': 1.48})

    def test_practice_bounds_need_scoreable_negatives_and_target_reaching_positives(self):
        rows = [{'clip': f'{label}_{index}', 'dtw_score': score, 'max_arm_elevation_deg': peak}
                for index in range(3)
                for label, score, peak in [('complete', 1, 90), ('incomplete', 8, 55)]]
        self.assertEqual(practice_bounds(rows, self.settings), (1, 8))
        rows[1]['dtw_score'] = None
        self.assertIsNone(practice_bounds(rows, self.settings))
        rows[1]['dtw_score'] = 8
        rows[0]['max_arm_elevation_deg'] = 60
        self.assertIsNone(practice_bounds(rows, self.settings))
