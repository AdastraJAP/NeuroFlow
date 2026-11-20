"""Configuration isolation tests: no MediaPipe/camera needed."""
import json
import tempfile
import unittest
from pathlib import Path

from config import LEGACY_VERSION, VERSION, load_settings, mode_dir, save_settings, set_position, set_recording_timing


class TestIndependentPositions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'settings.json'

    def test_initial_thresholds_unset(self):
        seated = load_settings(self.path)
        self.assertEqual(seated['position'], 'seated')
        self.assertIsNone(seated['reference_clip'])
        self.assertIsNone(seated['dtw_threshold'])
        standing = set_position('standing', self.path)
        self.assertIsNone(standing['reference_clip'])
        self.assertIsNone(standing['dtw_threshold'])

    def test_calibrations_do_not_leak_between_positions(self):
        seated = load_settings(self.path)
        seated['reference_clip'] = 'clips/seated/reference/ref_01.avi'
        seated['dtw_threshold'] = 5.5
        save_settings(seated, self.path)
        standing = set_position('standing', self.path)
        self.assertIsNone(standing['reference_clip'])
        self.assertIsNone(standing['dtw_threshold'])
        standing['reference_clip'] = 'clips/standing/reference/ref_01.avi'
        standing['dtw_threshold'] = 12.5
        save_settings(standing, self.path)
        seated_again = set_position('seated', self.path)
        self.assertEqual(seated_again['dtw_threshold'], 5.5)
        self.assertEqual(seated_again['reference_clip'], 'clips/seated/reference/ref_01.avi')
        self.assertEqual(set_position('standing', self.path)['dtw_threshold'], 12.5)

    def test_recording_timing_is_shared(self):
        s = set_recording_timing(8, 12, self.path)
        self.assertEqual((s['record_countdown_s'], s['rep_timeout_s']), (8, 12))
        s = set_position('standing', self.path)
        self.assertEqual((s['record_countdown_s'], s['rep_timeout_s']), (8, 12))

    def test_legacy_reference_is_reused_but_threshold_is_unset(self):
        self.path.write_text(json.dumps({'version': LEGACY_VERSION, 'conditions': {
            'seated': {'reference_clip': 'clips/seated/reference/ref_01.avi', 'dtw_threshold': 1.48},
            'standing': {'dtw_threshold': 2.5},
        }}))
        settings = load_settings(self.path)
        self.assertEqual(settings['version'], VERSION)
        self.assertEqual(settings['reference_clip'], 'clips/seated/reference/ref_01.avi')
        self.assertIsNone(settings['dtw_threshold'])
        self.assertEqual(mode_dir('traces', settings), Path('traces/seated/segmented_v1'))
        self.assertEqual(json.loads(self.path.read_text())['conditions']['seated']['dtw_threshold'], 1.48)
        self.assertIsNone(set_position('standing', self.path)['dtw_threshold'])

    def test_bad_timing_rejected(self):
        with self.assertRaises(ValueError):
            set_recording_timing(1, 10, self.path)
        with self.assertRaises(ValueError):
            set_recording_timing(5, 50, self.path)

    def test_save_preserves_other_condition(self):
        set_position('standing', self.path)
        s = load_settings(self.path)
        s['dtw_threshold'] = 7.2
        save_settings(s, self.path)
        raw = json.loads(self.path.read_text())
        self.assertIsNone(raw['conditions']['seated']['dtw_threshold'])
        self.assertEqual(raw['conditions']['standing']['dtw_threshold'], 7.2)


if __name__ == '__main__':
    unittest.main()
