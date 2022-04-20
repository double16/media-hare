import unittest
import tempfile
import os
import common


class EdlParseTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def test_parse_seconds(self):
        """
        Test that parsing an EDL with cuts using seconds works
        """
        fd, path = tempfile.mkstemp(suffix='.edl')
        os.write(fd, b"""## cuts using seconds
0.00	13.15	0
307.17	353.12	0
772.57	917.95	0
1284.78	1524.89	0
1689.12	1824.86	0
2473.24	2654.05	0
2965.80	3086.55	0
3592.42	3596.89	0
""")
        os.close(fd)
        events = common.parse_edl_cuts(path)
        os.remove(path)
        self.assertEqual(8, len(events))
        self.assertEqual(0.0, events[0].start)
        self.assertEqual(13.15, events[0].end)
        self.assertEqual(common.EdlType.CUT, events[0].event_type)
        self.assertEqual(772.57, events[2].start)
        self.assertEqual(917.95, events[2].end)
        self.assertEqual(common.EdlType.CUT, events[2].event_type)

    def test_parse_mmss(self):
        """
        Test that parsing an EDL with time spec MM:SS.SSS
        """
        fd, path = tempfile.mkstemp(suffix='.edl')
        os.write(fd, b"""
3:45.678	 4:15	0
""")
        os.close(fd)
        events = common.parse_edl_cuts(path)
        os.remove(path)
        self.assertEqual(1, len(events))
        self.assertEqual(225.678, events[0].start)
        self.assertEqual(255, events[0].end)
        self.assertEqual(common.EdlType.CUT, events[0].event_type)

    def test_parse_hhmmss(self):
        """
        Test that parsing an EDL with time spec HH:MM:SS.SSS
        """
        fd, path = tempfile.mkstemp(suffix='.edl')
        os.write(fd, b"""
0:03:45.678	 1:04:15	0
""")
        os.close(fd)
        events = common.parse_edl_cuts(path)
        os.remove(path)
        self.assertEqual(1, len(events))
        self.assertEqual(225.678, events[0].start)
        self.assertEqual(3855, events[0].end)

    def test_parse_title(self):
        """
        Test parsing titles
        """
        fd, path = tempfile.mkstemp(suffix='.edl')
        os.write(fd, b"""## parse title test
0:03:45.67	 1:04:15	2    Chapter 1
1:13:45.67	 1:35:15	2    Chapter   2
""")
        os.close(fd)
        events = common.parse_edl(path)
        os.remove(path)
        self.assertEqual(2, len(events))
        self.assertEqual(225.67, events[0].start)
        self.assertEqual(3855, events[0].end)
        self.assertEqual(common.EdlType.SCENE, events[0].event_type)
        self.assertEqual("Chapter 1", events[0].title)
        self.assertEqual("Chapter   2", events[1].title)

    def test_s_to_ts(self):
        self.assertEqual("01:02:03.400", common.s_to_ts(3723.4))
