#!/usr/bin/env python3
import logging
import unittest

from common import tools, proc_invoker


def _read_file(path):
    with open(path, 'rt') as f:
        result = f.read()
    f.close()
    return result


class ToolsTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        tools.mock_all()

    def tearDown(self) -> None:
        tools.mock_verify_all()

    def test_get_audio_layouts_ffmpeg_45(self):
        for layout_basename in ['ffmpeg-4-layouts.txt', 'ffmpeg-5-layouts.txt']:
            with self.subTest():
                tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
                    {'method_name': 'check_output', 'result': _read_file(f'../fixtures/{layout_basename}')},
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

    def test_get_audio_layouts_ffmpeg_6(self):
        for layout_basename in ['ffmpeg-6-layouts.txt']:
            with self.subTest():
                tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
                    {'method_name': 'check_output', 'result': _read_file(f'../fixtures/{layout_basename}')},
                ])
                layouts = tools.get_audio_layouts(refresh=True)
                self.assertEqual(28, len(layouts))
                layout_51 = list(filter(lambda e: e.name == '5.1', layouts))[0]
                self.assertIsNotNone(layout_51, '5.1 present')
                self.assertEqual(['FL', 'FR', 'FC', 'LFE', 'BL', 'BR'], layout_51.channels)
                self.assertEqual(['FL', 'FR', 'FC'], layout_51.voice_channels)

                layout_51side = list(filter(lambda e: e.name == '5.1(side)', layouts))[0]
                self.assertIsNotNone(layout_51side, '5.1(side) present')
                self.assertEqual(['FL', 'FR', 'FC', 'LFE', 'SL', 'SR'], layout_51side.channels)
                self.assertEqual(['FL', 'FR', 'FC'], layout_51side.voice_channels)


class AudioLayoutTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def setUp(self) -> None:
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            {'method_name': 'check_output', 'result': _read_file(f'../fixtures/ffmpeg-5-layouts.txt')},
        ])
        self.layouts = tools.get_audio_layouts(refresh=True)

    def test_get_audio_layouts_map_51_to_51(self):
        layout51 = tools.get_audio_layout_by_name('5.1')
        mapping = layout51.map_to(layout51)
        self.assertEqual({'FC': ['FC'], 'FL': ['FL'], 'FR': ['FR'], 'LFE': ['LFE'], 'BL': ['BL'], 'BR': ['BR']},
                         mapping)

    def test_get_audio_layouts_map_51side_to_51(self):
        layout51 = tools.get_audio_layout_by_name('5.1')
        layout51side = tools.get_audio_layout_by_name('5.1(side)')
        mapping = layout51side.map_to(layout51)
        self.assertEqual({'FC': ['FC'], 'FL': ['FL'], 'FR': ['FR'], 'LFE': ['LFE'], 'BL': ['SL'], 'BR': ['SR']},
                         mapping)

    def test_get_audio_layouts_map_51_to_51side(self):
        layout51 = tools.get_audio_layout_by_name('5.1')
        layout51side = tools.get_audio_layout_by_name('5.1(side)')
        mapping = layout51.map_to(layout51side)
        self.assertEqual({'FC': ['FC'], 'FL': ['FL'], 'FR': ['FR'], 'LFE': ['LFE'], 'SL': ['BL'], 'SR': ['BR']},
                         mapping)

    def test_get_audio_layouts_map_71wide_to_71(self):
        layout71 = tools.get_audio_layout_by_name('7.1')
        layout71wide = tools.get_audio_layout_by_name('7.1(wide)')
        mapping = layout71wide.map_to(layout71)
        self.assertEqual(
            {'FC': ['FC'], 'FL': ['FL'], 'FR': ['FR'], 'LFE': ['LFE'], 'BL': ['BL'], 'BR': ['BR'], 'SL': ['FLC'],
             'SR': ['FRC']},
            mapping)

    def test_get_audio_layouts_map_71_to_71wide(self):
        layout71 = tools.get_audio_layout_by_name('7.1')
        layout71wide = tools.get_audio_layout_by_name('7.1(wide)')
        mapping = layout71.map_to(layout71wide)
        self.assertEqual(
            {'FC': ['FC'], 'FL': ['FL'], 'FR': ['FR'], 'LFE': ['LFE'], 'BL': ['BL'], 'BR': ['BR'], 'FLC': ['SL'],
             'FRC': ['SR']},
            mapping)

    def test_get_audio_layouts_map_40_to_51(self):
        layout51 = tools.get_audio_layout_by_name('5.1')
        layout40 = tools.get_audio_layout_by_name('4.0')
        mapping = layout40.map_to(layout51)
        self.assertEqual({'FC': ['FC'], 'FL': ['FL'], 'FR': ['FR'], 'LFE': [], 'BL': ['BC'], 'BR': ['BC']},
                         mapping)
