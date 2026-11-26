"""Run from project root: python -m unittest discover -s tests -v"""
import unittest
from geometry import arm_elevation, shoulder_line_tilt, tilt_change

class TestHipFreeGeometry(unittest.TestCase):
    def test_down_is_zero(self):
        self.assertAlmostEqual(arm_elevation((100,100),(100,200)),0)
    def test_horizontal_is_ninety_both_sides(self):
        self.assertAlmostEqual(arm_elevation((100,100),(200,100)),90)
        self.assertAlmostEqual(arm_elevation((100,100),(0,100)),90)
    def test_up_is_180(self):
        self.assertAlmostEqual(arm_elevation((100,100),(100,0)),180)
    def test_45(self):
        self.assertAlmostEqual(arm_elevation((100,100),(200,200)),45)
    def test_tilt_horizontal_zero(self):
        self.assertAlmostEqual(shoulder_line_tilt((10,50),(110,50)),0)
    def test_tilt_relative_start(self):
        baseline=shoulder_line_tilt((10,50),(110,50))
        current=shoulder_line_tilt((10,50),(110,60))
        self.assertAlmostEqual(tilt_change(current,baseline),5.710593,places=4)
    def test_no_fake_angle_when_coincident(self):
        self.assertIsNone(arm_elevation((100,100),(100,100)))

if __name__ == '__main__': unittest.main()
