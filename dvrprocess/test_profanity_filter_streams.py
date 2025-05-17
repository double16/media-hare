import json
import logging
import os
import shutil
import tempfile
import unittest
from typing import Union
from xml.etree import ElementTree as ET

import profanity_filter
from common import tools, proc_invoker, constants, config, is_ripped_from_media

#
# Testing scenarios:
#
# File states:
#   - A: unfiltered, original subtitle text
#   - B: unfiltered, original subtitle image
#   - C: unfiltered, original subtitle image and text
#   - D: unfiltered, no original subtitle
#   - E: unfiltered, original subtitle text, marked as skipped
#   - F: unfiltered, original subtitle image, marked as skipped
#   - G: unfiltered, original subtitle image and text, marked as skipped
#   - H: unfiltered, no original subtitle, marked as skipped
#   - I: filtered, original subtitle text, filtered subtitle
#   - J: filtered, original subtitle image, ocr subtitle text, filtered subtitle
#   - K: filtered, original subtitle text and image, filtered subtitle
#   - L: filtered, no original subtitle, transcribed subtitle text and words, filtered subtitle
#   - M: unfiltered, transcribed original subtitle, marked as skipped
#   - N: unfiltered, transcribed original subtitle and words, marked as skipped
#   - O: filtered, original subtitle text, transcribed subtitle text and words, filtered subtitle
#   - P: filtered, original subtitle image, ocr subtitle text, transcribed subtitle words, filtered subtitle
#   - Q: filtered, original subtitle text and image, transcribed subtitle words, filtered subtitle
#   - R: filtered, no original subtitle, transcribed subtitle text, filtered subtitle
#   - S: unfiltered, original subtitle image, ocr subtitle text, transcribed subtitle words
#   - T: unfiltered, original subtitle text and image, transcribed subtitle words
#
#  Apply filter:
#    x A -> I
#    x B -> J
#    x C -> K
#    x D -> L
#    x E -> E
#    x F -> F
#    x G -> G
#    x H -> N
#    x I -> O
#    x J -> P
#    x K -> Q
#    - L -> L
#    x M -> N
#    - N -> N
#    - O -> O
#    - P -> P
#    x Q -> Q
#    - R -> L
#    - S -> P
#    - T -> Q
#
#  Remove filter:
#    - A -> A, B -> B, C -> C, D -> D, E -> E, F -> F, G -> G, H -> H
#    x I -> A
#    - J -> C
#    - K -> C
#    - L -> N
#    - M -> M, N -> N
#    - O -> N
#    - P -> S
#    - Q -> T
#    - R -> M
#    - S -> S, T -> T
#
#  To extract .wav from .mkv
#      ffmpeg -ss 01:26:14 -i input.mkv -t 00:00:06 -ac 1 -map 0:1 filtered.wav
#

EMPTY_WAV: bytes = (
        b'RIFF' +
        b'\x24\x00\x00\x00' +  # Chunk Size (36 bytes after this)
        b'WAVE' +
        b'fmt ' +
        b'\x10\x00\x00\x00' +  # Subchunk 1 size (16 for PCM)
        b'\x01\x00' +  # Audio format (PCM)
        b'\x01\x00' +  # NumChannels (1 for mono)
        b'\x80\xBB\x00\x00' +  # Sample rate (48000 Hz)
        b'\x00\xEE\x02\x00' +  # Byte rate (48000 * 1 * 16 / 8 = 96000)
        b'\x02\x00' +  # Block align (1 * 16 / 8 = 2)
        b'\x10\x00' +  # Bits per sample (16)
        b'data' +
        b'\x00\x00\x00\x00'  # Subchunk 2 size (0 bytes, since it's empty)
)


class ProfanityFilterStreamsTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        tools.mock_all()
        # This method will unnecessarily fail for our test cases
        profanity_filter.words_in_dictionary_pct = lambda a, b, c: 100.00

    def tearDown(self) -> None:
        tools.mock_verify_all()

    def _read_file(self, path):
        with open(path, 'rt') as f:
            result = f.read()
        f.close()
        return result

    def _mock_ffprobe(self, input_info_basename: str, tags: dict[str, str] = None):
        input_info_path = f'../fixtures/{input_info_basename}'
        if tags:
            with open(input_info_path, "rt") as f:
                input_info_json = json.load(f)
            input_info_json["format"]["tags"].update(tags)
            fd, input_info_path = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            with open(input_info_path, "wt") as f:
                json.dump(input_info_json, f)

        tools.ffprobe = proc_invoker.MockProcInvoker('ffprobe', mocks=[
            {'method_name': 'check_output', 'result': self._read_file(input_info_path)},
            {'method_name': 'check_output', 'result': ''}  # closed captions check
        ])

    def _mock_ffmpeg_extract_subtitle_original(self, subtitle_basename):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertEqual(1, len(list(filter(lambda e: e == '-i', arguments))),
                             '1 inputs expected: ' + str(arguments))
            self.assertEqual(1, len(list(filter(lambda e: e == '-map', arguments))),
                             '1 mapped stream expected: ' + str(arguments))
            output = arguments[-1]
            shutil.copy(f"../fixtures/{subtitle_basename}", output)
            return 0

        return mock

    def _mock_ffmpeg_extract_subtitle_original_filtered(self, original_basename, filtered_basename):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertEqual(1, len(list(filter(lambda e: e == '-i', arguments))),
                             '1 inputs expected: ' + str(arguments))
            self.assertEqual(2, len(list(filter(lambda e: e == '-map', arguments))),
                             '2 mapped stream expected: ' + str(arguments))
            original_output = list(filter(lambda e: '.original.' in e, arguments))[0]
            filtered_output = list(filter(lambda e: '.filtered.' in e, arguments))[0]
            shutil.copy(f"../fixtures/{original_basename}", original_output)
            shutil.copy(f"../fixtures/{filtered_basename}", filtered_output)
            return 0

        return mock

    def _mock_ffmpeg_extract_subtitle_original_filtered_words(self, original_basename, filtered_basename,
                                                              words_basename):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertEqual(1, len(list(filter(lambda e: e == '-i', arguments))),
                             '1 inputs expected: ' + str(arguments))
            self.assertEqual(3, len(list(filter(lambda e: e == '-map', arguments))),
                             '3 mapped stream expected: ' + str(arguments))
            original_output = list(filter(lambda e: '.original.' in e, arguments))[0]
            filtered_output = list(filter(lambda e: '.filtered.' in e, arguments))[0]
            words_output = list(filter(lambda e: '.words.' in e, arguments))[0]
            shutil.copy(f"../fixtures/{original_basename}", original_output)
            shutil.copy(f"../fixtures/{filtered_basename}", filtered_output)
            shutil.copy(f"../fixtures/{words_basename}", words_output)
            return 0

        return mock

    def _mock_ffmpeg_extract_subtitle_original_words(self, words_basename):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertEqual(1, len(list(filter(lambda e: e == '-i', arguments))),
                             '1 inputs expected: ' + str(arguments))
            self.assertEqual(1, len(list(filter(lambda e: e == '-map', arguments))),
                             '1 mapped stream expected: ' + str(arguments))
            words_output = list(filter(lambda e: '.original.words.' in e, arguments))[0]
            shutil.copy(f"../fixtures/{words_basename}", words_output)
            return 0

        return mock

    def _mock_ffmpeg_extract_subtitle_filtered_words(self, filtered_basename, words_basename):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertEqual(1, len(list(filter(lambda e: e == '-i', arguments))),
                             '1 inputs expected: ' + str(arguments))
            self.assertEqual(2, len(list(filter(lambda e: e == '-map', arguments))),
                             '2 mapped stream expected: ' + str(arguments))
            filtered_output = list(filter(lambda e: '.filtered.' in e, arguments))[0]
            words_output = list(filter(lambda e: '.words.' in e, arguments))[0]
            shutil.copy(f"../fixtures/{filtered_basename}", filtered_output)
            shutil.copy(f"../fixtures/{words_basename}", words_output)
            return 0

        return mock

    def _mock_ffmpeg_extract_audio_for_transcribing(self, raw_audio_basename: [None, str]):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertTrue('-map' in arguments and 'wav' in arguments, arguments)
            audio_filename = arguments[-1]
            self.assertTrue(audio_filename)
            if raw_audio_basename is None:
                with open(audio_filename, "wb") as fd:
                    fd.write(EMPTY_WAV)
            else:
                shutil.copy(f"../fixtures/{raw_audio_basename}", audio_filename)

        return mock

    def _mock_ffmpeg_extract_subtitle_transcribe_check(self, srt_basename: [None, str]):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertTrue('-map' in arguments and 'srt' in arguments, arguments)
            self.assertTrue(arguments[-1] == '-', 'stream to stdout')
            with open(f"../fixtures/{srt_basename}", "rt") as f:
                subtitle = f.read()
            return subtitle

        return mock

    def _mock_ffmpeg_extract_dvdsub(self, method_name: str, arguments: list, **kwargs):
        self.assertTrue('-map' in arguments and 'dvdsub' in arguments, arguments)
        return 0

    def _mock_subtitle_edit(self, subtitle_basename):
        def mock(method_name: str, arguments: list, **kwargs):
            output = '.'.join(arguments[-1].split('.')[0:-1]) + '.srt'
            shutil.copy(f"../fixtures/{subtitle_basename}", output)
            return 0

        return mock

    def _mock_ffmpeg_create_with_filtered_streams(self, input_file_count: int, mapped_stream_count: [None, int] = None,
                                                  expected_args: [None, list[str]] = None):
        def mock(method_name: str, arguments: list, **kwargs):
            arguments_str = str(arguments)
            self.assertEqual(input_file_count, len(list(filter(lambda e: e == '-i', arguments))),
                             f'{input_file_count} inputs expected: ' + arguments_str)
            if mapped_stream_count is not None:
                self.assertEqual(mapped_stream_count, len(list(filter(lambda e: e == '-map', arguments))),
                                 f'{mapped_stream_count} mapped streams expected: ' + arguments_str)
            self.assertTrue("'-metadata:s:s:0', 'title=Filtered'" in arguments_str,
                            'Filtered subtitle expected: ' + arguments_str)
            self.assertTrue("'-metadata:s:s:1', 'title=Filtered Only'" in arguments_str,
                            'Filtered Only subtitle expected: ' + arguments_str)
            self.assertTrue("'-metadata:s:s:2', 'title=Original'" in arguments_str,
                            'Original subtitle expected: ' + arguments_str)
            self.assertTrue("'-metadata:s:s:3', 'title=Words'" in arguments_str,
                            'Words subtitle expected: ' + arguments_str)
            self.assertTrue("'-metadata:s:a:0', 'title=Filtered'" in arguments_str,
                            'Filtered audio expected: ' + arguments_str)
            self.assertTrue("'-metadata:s:a:1', 'title=Original'" in arguments_str,
                            'Original audio expected: ' + arguments_str)
            self.assertTrue("':volume=0" in arguments_str, 'audio filter with mute expected: ' + arguments_str)
            self.assertTrue(
                f"{constants.K_AUDIO_TO_TEXT_VERSION}={profanity_filter.AUDIO_TO_TEXT_VERSION}" in arguments_str,
                'audio2text version expected: ' + arguments_str)
            self.assertTrue(
                f"{constants.K_FILTER_VERSION}={profanity_filter.FILTER_VERSION}" in arguments_str,
                'filter version expected: ' + arguments_str)
            self.assertTrue(
                f"{constants.K_FILTER_HASH}={profanity_filter.compute_filter_hash()}" in arguments_str,
                'filter hash expected: ' + arguments_str)
            stream_counts = [('Filtered', 2), ('Filtered Only', 1), ('Original', 2), ('Words', 1)]
            for title, count in stream_counts:
                self.assertEqual(count, len(list(filter(lambda e: e == f'title={title}', arguments))),
                                 f'{count} {title} streams expected: ' + arguments_str)

            if expected_args is not None:
                for expected_arg in expected_args:
                    self.assertTrue(expected_arg in arguments_str, 'expected: ' + expected_arg + ' in ' + arguments_str)

            output = arguments[-1]
            if output[0] == '/':
                with open(output, 'w') as f:
                    f.write('test')

            return 0

        return mock

    def _mock_ffmpeg_create_with_nofiltered_streams(self, input_file_count: int,
                                                    mapped_stream_count: [None, int] = None, removed=False):
        def mock(method_name: str, arguments: list, **kwargs):
            arguments_str = str(arguments)
            self.assertEqual(input_file_count, len(list(filter(lambda e: e == '-i', arguments))),
                             f'{input_file_count} inputs expected: ' + arguments_str)
            if mapped_stream_count is not None:
                self.assertEqual(mapped_stream_count, len(list(filter(lambda e: e == '-map', arguments))),
                                 f'{mapped_stream_count} mapped streams expected: ' + arguments_str)
            self.assertTrue("'-metadata:s:s:0', 'title=Original'" in arguments_str,
                            'Original subtitle expected: ' + arguments_str)
            self.assertTrue("'-metadata:s:s:1', 'title=Words'" in arguments_str,
                            'Words subtitle expected: ' + arguments_str)
            self.assertFalse("'title=Filtered'" in arguments_str, 'Filtered streams not expected: ' + arguments_str)
            self.assertFalse("'title=Filtered Only'" in arguments_str,
                             'Filtered Only streams not expected: ' + arguments_str)
            self.assertFalse("':volume=0" in arguments_str, 'audio filter with mute unexpected: ' + arguments_str)
            self.assertTrue(
                f"{constants.K_AUDIO_TO_TEXT_VERSION}={profanity_filter.AUDIO_TO_TEXT_VERSION}" in arguments_str,
                'audio2text version expected: ' + arguments_str)
            if removed:
                self.assertTrue(
                    f"{constants.K_FILTER_VERSION}='" in arguments_str,
                    'filter version not expected: ' + arguments_str)
                self.assertTrue(
                    f"{constants.K_FILTER_HASH}='" in arguments_str,
                    'filter hash not expected: ' + arguments_str)
            else:
                self.assertTrue(
                    f"{constants.K_FILTER_VERSION}={profanity_filter.FILTER_VERSION}" in arguments_str,
                    'filter version expected: ' + arguments_str)
                self.assertTrue(
                    f"{constants.K_FILTER_HASH}={profanity_filter.compute_filter_hash()}" in arguments_str,
                    'filter hash expected: ' + arguments_str)
            stream_counts = [('Filtered', 0), ('Filtered Only', 0), ('Original', 2), ('Words', 1)]
            for title, count in stream_counts:
                self.assertEqual(count, len(list(filter(lambda e: e == f'title={title}', arguments))),
                                 f'{count} {title} streams expected: ' + arguments_str)

            output = arguments[-1]
            if output[0] == '/':
                with open(output, 'w') as f:
                    f.write('test')

            return 0

        return mock

    def _mock_mkvpropedit_tags(self, expected_tags: dict[str, Union[None, str]] = None):
        def mock(method_name: str, arguments: list, **kwargs):
            if expected_tags is not None:
                global_tag_filename = None
                for arg in arguments:
                    if 'global:' in arg:
                        global_tag_filename = arg[7:]
                self.assertTrue(global_tag_filename, "Expected mkvpropedit --tags global:...")
                self.assertTrue(os.path.exists(global_tag_filename), 'tag filename exists')
                tree = ET.parse(global_tag_filename)
                root = tree.getroot()
                for k, v in expected_tags.items():
                    found = False
                    for el in root.findall('.//Simple'):
                        if el.find('Name').text == k:
                            found = True
                            if v is not None:
                                self.assertEqual(v, el.find('String').text, f"Tag {k} value")
                    self.assertTrue(found, f"Tag not found: {k}")
            return 0

        return mock

    def test_apply_unfiltered_sub_orig_text__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original('needs_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4, mapped_stream_count=8),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_text__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("not_filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original('not_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_nofiltered_streams(4, mapped_stream_count=5),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_image__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_image.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_dvdsub,
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_create_with_filtered_streams(5, mapped_stream_count=9),
        ])
        tools.subtitle_edit = proc_invoker.MockProcInvoker('subtitle-edit', mocks=[
            self._mock_subtitle_edit('needs_filtered.srt.txt')
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_image__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_image.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_dvdsub,
            self._mock_ffmpeg_extract_audio_for_transcribing("not_filtered.wav"),
            self._mock_ffmpeg_create_with_nofiltered_streams(5, mapped_stream_count=6),
        ])
        tools.subtitle_edit = proc_invoker.MockProcInvoker('subtitle-edit', mocks=[
            self._mock_subtitle_edit('not_filtered.srt.txt')
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_image_text__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_image_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original('needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4, mapped_stream_count=9),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_image_text__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_image_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("not_filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original('not_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_nofiltered_streams(4, mapped_stream_count=6),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_none__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_none.json',
                           {
                               constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION),
                               constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION),
                           })
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_create_with_filtered_streams(
                5, mapped_stream_count=8,
                expected_args=[
                    f"'-metadata:s:s:2', '{constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION}={profanity_filter.AUDIO_TO_TEXT_SUBTITLE_VERSION}'",
                ]),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_none__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_none.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("not_filtered.wav"),
            self._mock_ffmpeg_create_with_nofiltered_streams(5, mapped_stream_count=5),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_text_skipped__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_text_skipped.json')
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_image_skipped__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_image_skipped.json')
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_image_text_skipped__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_image_text_skipped.json')
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_none_skipped__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_none_skipped.json')
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_none__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_none.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original_filtered('needs_filtered.ssa.txt', 'needs_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4, mapped_stream_count=8),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_orig_image_sub_filtered_sub_words_none__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_image_sub_filtered_sub_words_none.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original_filtered('needs_filtered.srt.txt', 'needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4, mapped_stream_count=9),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_orig_transcribed_skipped__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_transcribed_skipped.json')
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_text__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {
                               constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION),
                               constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION),
                           })
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_subtitle_original_filtered_words('needs_filtered.ssa.txt',
                                                                       'needs_filtered.ssa.txt',
                                                                       'needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(3, mapped_stream_count=8),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_remove_filtered_sub_orig_text_sub_filtered_sub_words_text(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {
                               constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION),
                               constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION),
                           })
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_create_with_nofiltered_streams(1, mapped_stream_count=5, removed=True)
        ])
        profanity_filter.do_profanity_filter(mkv_path, filter_skip=True)

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_text__filter_hash_not_changed(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION),
                            constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION),
                            constants.K_FILTER_HASH: profanity_filter.compute_filter_hash()})
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_text__filter_version_changed(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {
                               constants.K_FILTER_VERSION: '0',
                               constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION),
                           })
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_subtitle_original_filtered_words('needs_filtered.ssa.txt',
                                                                       'needs_filtered.ssa.txt',
                                                                       'needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(3, mapped_stream_count=8),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_text__audio2text_version_changed(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {constants.K_AUDIO_TO_TEXT_VERSION: '0',
                            constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION),
                            constants.K_FILTER_HASH: profanity_filter.compute_filter_hash()})
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original_filtered('needs_filtered.ssa.txt', 'needs_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4, mapped_stream_count=8),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_text__audio2text_version_unchanged(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION)})
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_subtitle_original_filtered_words('needs_filtered.ssa.txt',
                                                                       'needs_filtered.ssa.txt',
                                                                       'needs_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(3, mapped_stream_count=8),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_transcribed_text_sub_filtered_sub_words_text__audio2text_subtitle_version_changed(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_transcribed_text_sub_filtered_sub_words_text.json',
                           {constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION),
                            constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION),
                            constants.K_FILTER_HASH: profanity_filter.compute_filter_hash()})
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_subtitle_original_words('needs_filtered.srt.txt'),
            self._mock_ffmpeg_extract_subtitle_filtered_words('needs_filtered.srt.txt',
                                                              'needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(
                4, mapped_stream_count=9,
                expected_args=[
                    f"{constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION}={profanity_filter.AUDIO_TO_TEXT_SUBTITLE_VERSION}"]),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_transcribed__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_transcribed_noversion.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original('needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4, mapped_stream_count=8),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_transcribed__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_transcribed_noversion.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("not_filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original('not_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_nofiltered_streams(4, mapped_stream_count=5),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_none__no_subtitle_generated(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_none.json',
                           {constants.K_AUDIO_TO_TEXT_VERSION: '0',
                            constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION),
                            constants.K_FILTER_HASH: profanity_filter.compute_filter_hash()})
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing(None),
        ])
        tools.mkvpropedit = proc_invoker.MockProcInvoker('mkvpropedit', mocks=[
            self._mock_mkvpropedit_tags(
                expected_tags={constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION)})
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_none_audio_orig_none__no_subtitle_generated(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_none_audio_none.json')
        tools.mkvpropedit = proc_invoker.MockProcInvoker('mkvpropedit', mocks=[
            self._mock_mkvpropedit_tags()
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_text__audio2text_version_greater_than_current(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION + 1),
                            constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION),
                            constants.K_FILTER_HASH: profanity_filter.compute_filter_hash()})
        self.assertEqual(profanity_filter.CMD_RESULT_UNCHANGED, profanity_filter.do_profanity_filter(mkv_path),
                         "return code")

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_text__audio2text_version_greater_than_current_forced(
            self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION + 1),
                            constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION),
                            constants.K_FILTER_HASH: profanity_filter.compute_filter_hash()})
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original_filtered('needs_filtered.ssa.txt',
                                                                 'needs_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4, mapped_stream_count=8),
        ])
        self.assertEqual(profanity_filter.CMD_RESULT_FILTERED,
                         profanity_filter.do_profanity_filter(mkv_path, force=True), "return code")

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_text__filter_version_greater_than_current(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION),
                            constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION + 1),
                            constants.K_FILTER_HASH: ''})
        self.assertEqual(profanity_filter.CMD_RESULT_UNCHANGED, profanity_filter.do_profanity_filter(mkv_path),
                         "return code")

    def test_apply_filtered_sub_orig_text_sub_filtered_sub_words_text__filter_version_greater_than_current_forced(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json',
                           {constants.K_AUDIO_TO_TEXT_VERSION: str(profanity_filter.AUDIO_TO_TEXT_VERSION),
                            constants.K_FILTER_VERSION: str(profanity_filter.FILTER_VERSION + 1),
                            constants.K_FILTER_HASH: ''})
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing(None),
            self._mock_ffmpeg_extract_subtitle_original_filtered_words('needs_filtered.ssa.txt',
                                                                       'needs_filtered.ssa.txt',
                                                                       'needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(3, mapped_stream_count=8),
        ])
        self.assertEqual(profanity_filter.CMD_RESULT_FILTERED,
                         profanity_filter.do_profanity_filter(mkv_path, force=True), "return code")

    def test_apply_unfiltered_sub_orig_text__mute_voice_channels__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            self._mock_ffmpeg_extract_subtitle_original('needs_filtered.ssa.txt'),
            {'method_name': 'check_output', 'result': self._read_file('../fixtures/ffmpeg-5-layouts.txt')},
            self._mock_ffmpeg_create_with_filtered_streams(4, mapped_stream_count=8),
        ])
        profanity_filter.do_profanity_filter(mkv_path, mute_channels=config.MuteChannels.VOICE)

    def test_apply_unfiltered_sub_orig_image__mute_voice_channels__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_image.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_dvdsub,
            self._mock_ffmpeg_extract_audio_for_transcribing("filtered.wav"),
            {'method_name': 'check_output', 'result': self._read_file('../fixtures/ffmpeg-5-layouts.txt')},
            self._mock_ffmpeg_create_with_filtered_streams(5, mapped_stream_count=9, expected_args=['pan=5.1']),
        ])
        tools.subtitle_edit = proc_invoker.MockProcInvoker('subtitle-edit', mocks=[
            self._mock_subtitle_edit('needs_filtered.srt.txt')
        ])
        profanity_filter.do_profanity_filter(mkv_path, mute_channels=config.MuteChannels.VOICE)

    def test_is_ripped_from_media_true(self):
        with open("../fixtures/media_state_ripped.json", "rt") as f:
            input_info_json = json.load(f)
        self.assertTrue(is_ripped_from_media(input_info_json))

    def test_is_ripped_from_media_false(self):
        with open("../fixtures/media_state_unfiltered_sub_orig_image.json", "rt") as f:
            input_info_json = json.load(f)
        self.assertFalse(is_ripped_from_media(input_info_json))
