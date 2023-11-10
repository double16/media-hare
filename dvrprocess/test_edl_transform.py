import logging
import tempfile
import unittest
from pathlib import Path

import edl_normalize


class EdlTransformTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def test_normalize(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.edl') as dest:
            dest.file.close()
            edl_normalize.edl_normalize('../fixtures/edl_s.edl', dest.name)
            self.assertEqual(Path('../fixtures/edl_ts.edl').read_text(), Path(dest.name).read_text())

    def test_simplify(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.edl') as dest:
            dest.file.close()
            edl_normalize.edl_simplify('../fixtures/edl_ts.edl', dest.name)
            self.assertEqual(Path('../fixtures/edl_s.edl').read_text(), Path(dest.name).read_text())


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
