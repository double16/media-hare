import os
import tempfile
import unittest

import common
from common import edl_util


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
        events = edl_util.parse_edl_cuts(path)
        os.remove(path)
        self.assertEqual(8, len(events))
        self.assertEqual(0.0, events[0].start)
        self.assertEqual(13.15, events[0].end)
        self.assertEqual(edl_util.EdlType.CUT, events[0].event_type)
        self.assertEqual(772.57, events[2].start)
        self.assertEqual(917.95, events[2].end)
        self.assertEqual(edl_util.EdlType.CUT, events[2].event_type)

    def test_parse_mmss(self):
        """
        Test that parsing an EDL with time spec MM:SS.SSS
        """
        fd, path = tempfile.mkstemp(suffix='.edl')
        os.write(fd, b"""
3:45.678	 4:15	0
""")
        os.close(fd)
        events = edl_util.parse_edl_cuts(path)
        os.remove(path)
        self.assertEqual(1, len(events))
        self.assertEqual(225.678, events[0].start)
        self.assertEqual(255, events[0].end)
        self.assertEqual(edl_util.EdlType.CUT, events[0].event_type)

    def test_parse_hhmmss(self):
        """
        Test that parsing an EDL with time spec HH:MM:SS.SSS
        """
        fd, path = tempfile.mkstemp(suffix='.edl')
        os.write(fd, b"""
0:03:45.678	 1:04:15	0
""")
        os.close(fd)
        events = edl_util.parse_edl_cuts(path)
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
        events = edl_util.parse_edl(path)
        os.remove(path)
        self.assertEqual(2, len(events))
        self.assertEqual(225.67, events[0].start)
        self.assertEqual(3855, events[0].end)
        self.assertEqual(edl_util.EdlType.SCENE, events[0].event_type)
        self.assertEqual("Chapter 1", events[0].title)
        self.assertEqual("Chapter   2", events[1].title)

    def test_s_to_ts(self):
        self.assertEqual("01:02:03.400", common.s_to_ts(3723.4))

    def test_parse_symbolic(self):
        """
        Test that parsing an EDL with symbolic operations works
        """
        fd, path = tempfile.mkstemp(suffix='.edl')
        os.write(fd, b"""## cuts using seconds
0.00	13.15	cut
307.17	353.12	mute
772.57	917.95	scene
1284.78	1524.89	com
1689.12	1824.86	commercial
2473.24	2654.05	blur
""")
        os.close(fd)
        events = edl_util.parse_edl(path)
        os.remove(path)
        self.assertEqual(edl_util.EdlType.CUT, events[0].event_type)
        self.assertEqual(edl_util.EdlType.MUTE, events[1].event_type)
        self.assertEqual(edl_util.EdlType.SCENE, events[2].event_type)
        self.assertEqual(edl_util.EdlType.COMMERCIAL, events[3].event_type)
        self.assertEqual(edl_util.EdlType.COMMERCIAL, events[4].event_type)
        self.assertEqual(edl_util.EdlType.BACKGROUND_BLUR, events[5].event_type)

    def test_overlap_equal(self):
        a = edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL)
        b = edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL)
        self.assertEqual(300, a.get_overlap(b))

    def test_overlap_a_before_b(self):
        a = edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL)
        b = edl_util.EdlEvent(200, 300, edl_util.EdlType.COMMERCIAL)
        self.assertEqual(100, a.get_overlap(b))

    def test_overlap_b_before_a(self):
        a = edl_util.EdlEvent(200, 300, edl_util.EdlType.COMMERCIAL)
        b = edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL)
        self.assertEqual(100, a.get_overlap(b))

    def test_overlap_a_contains_b(self):
        a = edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL)
        b = edl_util.EdlEvent(100, 200, edl_util.EdlType.COMMERCIAL)
        self.assertEqual(100, a.get_overlap(b))
        self.assertEqual(100, b.get_overlap(a))

    def test_overlap_disjoint(self):
        a = edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL)
        b = edl_util.EdlEvent(400, 500, edl_util.EdlType.COMMERCIAL)
        self.assertEqual(0, a.get_overlap(b))
        self.assertEqual(0, b.get_overlap(a))

    def test_align_commercial_breaks_empty_or_none(self):
        self.assertEqual([], edl_util.align_commercial_breaks(None)[0])
        self.assertEqual([], edl_util.align_commercial_breaks([])[0])
        self.assertEqual([[]], edl_util.align_commercial_breaks([[]])[0])

    def validated_align_commercial_breaks(self, input: list[list[edl_util.EdlEvent]], output:list[list[edl_util.EdlEvent]] , combined: list[edl_util.EdlEvent]):
        self.assertEqual(len(input), len(output))
        for output_el in output:
            self.assertEqual(len(combined), len(output_el))

    def test_align_commercial_breaks_single(self):
        input = [[
            edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL),
            edl_util.EdlEvent(600, 900, edl_util.EdlType.COMMERCIAL),
        ]]
        output, score, combined = edl_util.align_commercial_breaks(input)
        self.validated_align_commercial_breaks(input, output, combined)
        self.assertEqual(input, output)
        self.assertEqual(0.001, score)

    def test_align_commercial_breaks_two_similar(self):
        input = [
            [
                edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL),
                edl_util.EdlEvent(600, 900, edl_util.EdlType.COMMERCIAL),
            ],
            [
                edl_util.EdlEvent(2, 303, edl_util.EdlType.COMMERCIAL),
                edl_util.EdlEvent(578, 800, edl_util.EdlType.COMMERCIAL),
            ],
        ]
        output, score, combined = edl_util.align_commercial_breaks(input)
        self.validated_align_commercial_breaks(input, output, combined)
        self.assertEqual(input, output)
        self.assertAlmostEqual(55.86, score, 2)

    def test_align_commercial_breaks_two_disjoint(self):
        input = [
            [
                edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL),
                edl_util.EdlEvent(600, 900, edl_util.EdlType.COMMERCIAL),
            ],
            [
                edl_util.EdlEvent(2, 150, edl_util.EdlType.COMMERCIAL),
                edl_util.EdlEvent(400, 500, edl_util.EdlType.COMMERCIAL),
            ],
        ]
        expected = [
            [
                edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL),
                None,
                edl_util.EdlEvent(600, 900, edl_util.EdlType.COMMERCIAL),
            ],
            [
                edl_util.EdlEvent(2, 150, edl_util.EdlType.COMMERCIAL),
                edl_util.EdlEvent(400, 500, edl_util.EdlType.COMMERCIAL),
                None,
            ],
        ]
        output, score, combined = edl_util.align_commercial_breaks(input)
        self.validated_align_commercial_breaks(input, output, combined)
        self.assertEqual(expected, output)
        self.assertAlmostEqual(109.48, score, 2)

    def test_align_commercial_breaks_three_disjoint(self):
        input = [
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
        expected = [
            [
                edl_util.EdlEvent(0, 300, edl_util.EdlType.COMMERCIAL),
                None,
                edl_util.EdlEvent(600, 900, edl_util.EdlType.COMMERCIAL),
            ],
            [
                edl_util.EdlEvent(2, 150, edl_util.EdlType.COMMERCIAL),
                edl_util.EdlEvent(400, 500, edl_util.EdlType.COMMERCIAL),
                None,
            ],
            [
                edl_util.EdlEvent(2, 152, edl_util.EdlType.COMMERCIAL),
                edl_util.EdlEvent(410, 460, edl_util.EdlType.COMMERCIAL),
                edl_util.EdlEvent(602, 880, edl_util.EdlType.COMMERCIAL),
            ],
        ]
        output, score, combined = edl_util.align_commercial_breaks(input)
        self.validated_align_commercial_breaks(input, output, combined)
        self.assertEqual(expected, output)
        self.assertAlmostEqual(140.10, score, 2)

    def test_align_commercial_breaks_three_equal(self):
        input = [
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
        output, score, combined = edl_util.align_commercial_breaks(input)
        self.validated_align_commercial_breaks(input, output, combined)
        self.assertEqual(input, output)
        self.assertEqual(0.001, score)

    def test_align_commercial_breaks_s02(self):
        input = []
        for root, dirs, files in os.walk('../fixtures/combreaks/one'):
            for file in files:
                if file.endswith(".edl"):
                    _, commercial_breaks, _ = edl_util.parse_commercials(os.path.join(root, file), 3600)
                    input.append(commercial_breaks)
        output, score, combined = edl_util.align_commercial_breaks(input)
        self.validated_align_commercial_breaks(input, output, combined)
        self.assertAlmostEqual(595.49, score, 2)
