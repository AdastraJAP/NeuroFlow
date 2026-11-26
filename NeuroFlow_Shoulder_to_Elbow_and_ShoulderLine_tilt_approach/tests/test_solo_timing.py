import unittest
from solo_timing import SoloTimer


class TestSoloTimer(unittest.TestCase):
    def test_countdown_then_automatic_stop(self):
        timer = SoloTimer(5, 10)
        timer.arm(100.0)
        self.assertEqual(timer.countdown_left(100), 5)
        self.assertIsNone(timer.tick(104.99))
        self.assertEqual(timer.state, 'countdown')
        self.assertEqual(timer.tick(105.01), 'started')
        self.assertEqual(timer.state, 'recording')
        self.assertIsNone(timer.tick(114.99))
        self.assertEqual(timer.tick(115.01), 'finished')
        self.assertEqual(timer.state, 'finished')

    def test_cancel_during_countdown(self):
        timer = SoloTimer(5, 10)
        timer.arm(100)
        timer.cancel()
        self.assertIsNone(timer.tick(200))
        self.assertEqual(timer.state, 'cancelled')

    def test_no_double_start(self):
        timer = SoloTimer(5, 10)
        timer.arm(100)
        with self.assertRaises(ValueError):
            timer.arm(101)


if __name__ == '__main__':
    unittest.main()
