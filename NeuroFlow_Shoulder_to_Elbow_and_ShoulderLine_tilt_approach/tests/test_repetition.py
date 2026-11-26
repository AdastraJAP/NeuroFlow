import unittest

from repetition import RepetitionDetector, detect_repetition, resample


def at_intervals(values):
    return [(index * 100, value) for index, value in enumerate(values)]


class TestRepetition(unittest.TestCase):
    def test_hysteresis_and_dwell_ignore_brief_start_and_return(self):
        samples = at_intervals([5] * 6 + [22, 5, 19, 22, 44, 90, 11, 50, 14, 12] + [8] * 4)
        rep = detect_repetition(samples, 25)
        self.assertEqual((rep.outcome, rep.start_ms, rep.end_ms), ('complete', 900, 1500))
        self.assertEqual(rep.segment[-1], (1500, 12))
        self.assertEqual(max(angle for _, angle in rep.segment), 90)

    def test_slow_repetition_exceeds_legacy_ten_seconds(self):
        samples = at_intervals([6] * 6 + [24, 40] + [90] * 120 + [65, 40] + [6] * 5)
        rep = detect_repetition(samples, 25)
        self.assertEqual(rep.outcome, 'complete')
        self.assertGreater(rep.end_ms - rep.start_ms, 10_000)

    def test_timeout_and_clip_ending_during_return_dwell(self):
        samples = at_intervals([5] * 6 + [30, 60, 90] + [90] * 65)
        self.assertEqual(detect_repetition(samples, 6).outcome, 'incomplete_return')
        almost = samples[:15] + [(1500, 8), (1600, 7)]
        self.assertEqual(detect_repetition(almost, 6).outcome, 'incomplete_return')

    def test_tracking_loss_abstains_even_if_arm_returns_later(self):
        samples = at_intervals([5] * 6 + [30, 60, 90, None, 10] + [6] * 10)
        rep = detect_repetition(samples, 25)
        self.assertEqual(rep.outcome, 'unable_to_judge')
        self.assertFalse(rep.segment)

    def test_missing_before_rest_can_recover(self):
        samples = at_intervals([None, 5, None] + [5] * 6 + [25, 65, 90] + [5] * 5)
        self.assertEqual(detect_repetition(samples, 25).outcome, 'complete')

    def test_camera_gap_or_duplicate_timestamp_abstains(self):
        rest = at_intervals([5] * 6)
        self.assertEqual(detect_repetition(rest + [(850, 80)], 25).outcome, 'unable_to_judge')
        self.assertEqual(detect_repetition(rest + [(500, 80)], 25).outcome, 'unable_to_judge')

    def test_no_movement_differs_from_missing_baseline(self):
        self.assertEqual(detect_repetition(at_intervals([5] * 90), 25).outcome, 'no_start')
        self.assertEqual(detect_repetition(at_intervals([45] * 90), 25).outcome, 'unable_to_judge')

    def test_incremental_detection_matches_offline_and_ignores_trailing_frames(self):
        samples = at_intervals([6] * 6 + [24, 40, 90] + [6] * 5 + [None, 90])
        detector = RepetitionDetector(25)
        for timestamp, angle in samples:
            detector.push(timestamp, angle)
        self.assertEqual(detector.finish(), detect_repetition(samples, 25))
        self.assertEqual(detector.finish().end_ms, 900)

    def test_resampling_respects_irregular_time(self):
        self.assertEqual(resample([(0, 0), (100, 20), (500, 60)], 6), [0, 20, 30, 40, 50, 60])
