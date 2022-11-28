#!/usr/bin/env python3
import logging
import unittest

from common import tools, proc_invoker


class ToolsTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        tools.mock_all()

    def tearDown(self) -> None:
        tools.mock_verify_all()

    def _read_file(self, path):
        with open(path, 'rt') as f:
            result = f.read()
        f.close()
        return result

    def test_get_audio_layouts_ffmpeg5(self):
        for layout_basename in ['ffmpeg-4-layouts.txt', 'ffmpeg-5-layouts.txt']:
            with self.subTest():
                tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
                    {'method_name': 'check_output', 'result': self._read_file(f'../fixtures/{layout_basename}')},
                ])
                layouts = tools.get_audio_layouts(refresh=True)
                self.assertEqual(26, len(layouts))
                layout_51 = list(filter(lambda e: e.name == '5.1', layouts))[0]
                self.assertIsNotNone(layout_51, '5.1 present')
                self.assertEqual(['FL', 'FR', 'FC', 'LFE', 'BL', 'BR'], layout_51.channels)
                self.assertEqual(['FL', 'FR', 'FC'], layout_51.voice_channels)

                layout_51side = list(filter(lambda e: e.name == '5.1(side)', layouts))[0]
                self.assertIsNotNone(layout_51side, '5.1(side) present')
                self.assertEqual(['FL', 'FR', 'FC', 'LFE', 'SL', 'SR'], layout_51side.channels)
                self.assertEqual(['FL', 'FR', 'FC'], layout_51side.voice_channels)
