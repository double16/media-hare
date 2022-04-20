import unittest
import common


class FindDesiredKeyframeTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def test_empty(self):
        self.assertEqual(100.0, common.find_desired_keyframe([], 100.0))

    def test_single_entry(self):
        l = [100.0]
        self.assertEqual(100.0, common.find_desired_keyframe(l, 100.0))
        self.assertEqual(100.0, common.find_desired_keyframe(l, 90.0))
        self.assertEqual(100.0, common.find_desired_keyframe(l, 110.0))

    def test_single_two(self):
        l = [50.5, 100.0]
        self.assertEqual(100.0, common.find_desired_keyframe(l, 100.0))
        self.assertEqual(100.0, common.find_desired_keyframe(l, 90.0))
        self.assertEqual(100.0, common.find_desired_keyframe(l, 110.0))

    def test_exact(self):
        l = [25.0, 50.5, 100.0, 201.2]
        self.assertEqual(25.0, common.find_desired_keyframe(l, 25.0))
        self.assertEqual(50.5, common.find_desired_keyframe(l, 50.5))
        self.assertEqual(100.0, common.find_desired_keyframe(l, 100.0))
        self.assertEqual(201.2, common.find_desired_keyframe(l, 201.2))

    def test_closest(self):
        l = [25.0, 50.5, 100.0, 201.2]
        self.assertEqual(25.0, common.find_desired_keyframe(l, 10.0))
        self.assertEqual(25.0, common.find_desired_keyframe(l, 21.0))
        self.assertEqual(25.0, common.find_desired_keyframe(l, 27.0))
        self.assertEqual(50.5, common.find_desired_keyframe(l, 48.0))
        self.assertEqual(50.5, common.find_desired_keyframe(l, 51.0))
        self.assertEqual(50.5, common.find_desired_keyframe(l, 60.0))
        self.assertEqual(100.0, common.find_desired_keyframe(l, 88.0))
        self.assertEqual(100.0, common.find_desired_keyframe(l, 150.0))
        self.assertEqual(201.2, common.find_desired_keyframe(l, 220.0))

    def test_force_after(self):
        l = [25.0, 50.5, 100.0, 201.2]
        self.assertEqual(25.0, common.find_desired_keyframe(l, 10.0, True))
        self.assertEqual(25.0, common.find_desired_keyframe(l, 21.0, True))
        self.assertEqual(25.0, common.find_desired_keyframe(l, 27.0, True))
        self.assertEqual(50.5, common.find_desired_keyframe(l, 48.0, True))
        self.assertEqual(50.5, common.find_desired_keyframe(l, 51.0, True))
        self.assertEqual(50.5, common.find_desired_keyframe(l, 60.0, True))
        self.assertEqual(100.0, common.find_desired_keyframe(l, 88.0, True))
        self.assertEqual(100.0, common.find_desired_keyframe(l, 150.0, True))
        self.assertEqual(201.2, common.find_desired_keyframe(l, 220.0, True))
