import logging
import unittest
from common import progress


class ProgressReporterTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def test_new_progress(self):
        p = progress.progress('test', 0, 100)
        self.assertIsInstance(p, progress.Progress)

    def test_progress_cycle(self):
        p = progress.progress('test', 0, 100)
        p.progress(25)
        p.progress(50)
        p.progress(100)
        p.stop()
        self.assertTrue(True, 'no failures')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
