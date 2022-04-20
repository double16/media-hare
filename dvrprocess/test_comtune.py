import unittest
import comtune


class ComTuneTest(unittest.TestCase):
    def test_fitness_value_same_sigma(self):
        s1 = comtune.fitness_value(60, 0, 0)
        s2 = comtune.fitness_value(60, 0, 0)
        self.assertEqual(s1, s2)

    def test_fitness_value_different_sigma(self):
        s1 = comtune.fitness_value(30, 0, 0)
        s2 = comtune.fitness_value(60, 0, 0)
        self.assertGreater(s1, s2)

    def test_fitness_value_different_duration(self):
        s1 = comtune.fitness_value(60, 2, 0)
        s2 = comtune.fitness_value(60, 10, 0)
        self.assertGreater(s1, s2)

    def test_fitness_value_different_default_count(self):
        s1 = comtune.fitness_value(60, 10, 1)
        s2 = comtune.fitness_value(60, 10, 10)
        self.assertGreater(s1, s2)

    def test_fitness_value_prioritize_average(self):
        s1 = comtune.fitness_value(60, 2, 1)
        s2 = comtune.fitness_value(5, 14, 1)
        self.assertGreater(s1, s2)
        s1 = comtune.fitness_value(60, 2, 10)
        s2 = comtune.fitness_value(5, 14, 1)
        self.assertGreater(s1, s2)

    def test_fitness_value_prioritize_sigma(self):
        s1 = comtune.fitness_value(40, 2, 10)
        s2 = comtune.fitness_value(60, 2, 10)
        self.assertGreater(s1, s2)
        s1 = comtune.fitness_value(40, 2, 1)
        s2 = comtune.fitness_value(60, 2, 10)
        self.assertGreater(s1, s2)


if __name__ == '__main__':
    unittest.main()
