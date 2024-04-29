import unittest

import common


class FindDesiredKeyframeTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def test_empty(self):
        self.assertEqual(100.0, common.find_desired_keyframe([], 100.0))

    def test_single_entry(self):
        keyframes = [0.0, 100.0]
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 100.0))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 90.0))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 110.0))

    def test_single_two(self):
        keyframes = [0.0, 50.5, 100.0]
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 100.0))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 90.0))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 110.0))

    def test_exact(self):
        keyframes = [0.0, 25.0, 50.5, 100.0, 201.2]
        self.assertEqual(25.0, common.find_desired_keyframe(keyframes, 25.0))
        self.assertEqual(50.5, common.find_desired_keyframe(keyframes, 50.5))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 100.0))
        self.assertEqual(201.2, common.find_desired_keyframe(keyframes, 201.2))

    def test_closest(self):
        keyframes = [0.0, 25.0, 50.5, 100.0, 201.2]
        self.assertEqual(0.0, common.find_desired_keyframe(keyframes, 10.0))
        self.assertEqual(25.0, common.find_desired_keyframe(keyframes, 21.0))
        self.assertEqual(25.0, common.find_desired_keyframe(keyframes, 27.0))
        self.assertEqual(50.5, common.find_desired_keyframe(keyframes, 48.0))
        self.assertEqual(50.5, common.find_desired_keyframe(keyframes, 51.0))
        self.assertEqual(50.5, common.find_desired_keyframe(keyframes, 60.0))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 88.0))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 150.0))
        self.assertEqual(201.2, common.find_desired_keyframe(keyframes, 220.0))

    def test_closest_offset(self):
        keyframes = [1.0, 26.0, 51.5, 101.0, 202.2]
        self.assertEqual(1.0, common.find_desired_keyframe(keyframes, 10.0))
        self.assertEqual(26.0, common.find_desired_keyframe(keyframes, 21.0))
        self.assertEqual(26.0, common.find_desired_keyframe(keyframes, 27.0))
        self.assertEqual(51.5, common.find_desired_keyframe(keyframes, 48.0))
        self.assertEqual(51.5, common.find_desired_keyframe(keyframes, 51.0))
        self.assertEqual(51.5, common.find_desired_keyframe(keyframes, 60.0))
        self.assertEqual(101.0, common.find_desired_keyframe(keyframes, 88.0))
        self.assertEqual(101.0, common.find_desired_keyframe(keyframes, 150.0))
        self.assertEqual(202.2, common.find_desired_keyframe(keyframes, 220.0))

    def test_force_after(self):
        keyframes = [0.0, 25.0, 50.5, 100.0, 201.2]
        self.assertEqual(25.0, common.find_desired_keyframe(keyframes, 10.0, common.KeyframeSearchPreference.AFTER))
        self.assertEqual(25.0, common.find_desired_keyframe(keyframes, 21.0, common.KeyframeSearchPreference.AFTER))
        self.assertEqual(50.5, common.find_desired_keyframe(keyframes, 27.0, common.KeyframeSearchPreference.AFTER))
        self.assertEqual(50.5, common.find_desired_keyframe(keyframes, 48.0, common.KeyframeSearchPreference.AFTER))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 51.0, common.KeyframeSearchPreference.AFTER))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 60.0, common.KeyframeSearchPreference.AFTER))
        self.assertEqual(100.0, common.find_desired_keyframe(keyframes, 88.0, common.KeyframeSearchPreference.AFTER))
        self.assertEqual(201.2, common.find_desired_keyframe(keyframes, 150.0, common.KeyframeSearchPreference.AFTER))
        self.assertEqual(201.2, common.find_desired_keyframe(keyframes, 220.0, common.KeyframeSearchPreference.AFTER))
