import unittest
import comtune
from dvrprocess.common import edl_util


class ComTuneTest(unittest.TestCase):
    commercial_breaks_disjoint = [
        [
            edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(600, 900, edl_util.EdlType.COMMERCIAL),
        ],
        [
            edl_util.EdlEvent(2, 150, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(400, 500, edl_util.EdlType.COMMERCIAL),
        ],
        [
            edl_util.EdlEvent(2, 152, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(410, 460, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(602, 880, edl_util.EdlType.COMMERCIAL),
        ],
    ]
    commercial_breaks_perfect = [
        [
            edl_util.EdlEvent(2, 152, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(410, 460, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(602, 880, edl_util.EdlType.COMMERCIAL),
        ],
        [
            edl_util.EdlEvent(2, 152, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(410, 460, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(602, 880, edl_util.EdlType.COMMERCIAL),
        ],
        [
            edl_util.EdlEvent(2, 152, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(410, 460, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(602, 880, edl_util.EdlType.COMMERCIAL),
        ],
    ]

    def test_fitness_value_same_sigma(self):
        s1 = comtune.fitness_value(60, 0, 0)
        s2 = comtune.fitness_value(60, 0, 0)
        self.assertEqual(s1, s2)

    def test_fitness_value_different_sigma(self):
        s1 = comtune.fitness_value(30, 0, 0)
        s2 = comtune.fitness_value(60, 0, 0)
        self.assertGreater(s1, s2)

    def test_fitness_value_different_duration_under_min(self):
        s1 = comtune.fitness_value(60, 2, 0)
        s2 = comtune.fitness_value(60, 10, 0)
        self.assertEqual(s1, s2)

    def test_fitness_value_different_duration(self):
        s1 = comtune.fitness_value(60, 62, 0)
        s2 = comtune.fitness_value(60, 70, 0)
        self.assertGreater(s1, s2)

    def test_fitness_value_different_default_count(self):
        s1 = comtune.fitness_value(60, 70, 1)
        s2 = comtune.fitness_value(60, 70, 10)
        self.assertGreater(s1, s2)

    def test_fitness_value_prioritize_average(self):
        s1 = comtune.fitness_value(60, 62, 1)
        s2 = comtune.fitness_value(5, 74, 1)
        self.assertGreater(s1, s2)
        s1 = comtune.fitness_value(60, 62, 10)
        s2 = comtune.fitness_value(5, 74, 1)
        self.assertGreater(s1, s2)

    def test_fitness_value_prioritize_sigma(self):
        s1 = comtune.fitness_value(40, 62, 10)
        s2 = comtune.fitness_value(60, 62, 10)
        self.assertGreater(s1, s2)
        s1 = comtune.fitness_value(40, 62, 1)
        s2 = comtune.fitness_value(60, 62, 10)
        self.assertGreater(s1, s2)

    def test_fitness_value_commercial_breaks(self):
        s1 = comtune.fitness_value(40, 62, 10, commercial_breaks=self.commercial_breaks_perfect)
        s2 = comtune.fitness_value(40, 62, 10, commercial_breaks=self.commercial_breaks_disjoint)
        self.assertGreater(s1, s2)

    def test_fitness_value_prioritize_commercial_breaks(self):
        s1 = comtune.fitness_value(60, 74, 10, commercial_breaks=self.commercial_breaks_perfect)
        s2 = comtune.fitness_value(40, 62, 1, commercial_breaks=self.commercial_breaks_disjoint)
        self.assertGreater(s1, s2)


if __name__ == '__main__':
    unittest.main()
