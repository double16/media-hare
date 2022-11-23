import logging
import os
import shutil
import tempfile
import unittest

import profanity_filter
from common import tools, proc_invoker


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
#    - I -> A
#    - J -> C
#    - K -> C
#    - L -> N
#    - M -> M, N -> N
#    - O -> N
#    - P -> S
#    - Q -> T
#    - R -> M
#    - S -> S, T -> T


class ProfanityFilterStreamsTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg')
        tools.ffprobe = proc_invoker.MockProcInvoker('ffprobe')
        tools.comskip = proc_invoker.MockProcInvoker('comskip')
        tools.comskip_gui = proc_invoker.MockProcInvoker('comskip-gui')
        tools.mkvpropedit = proc_invoker.MockProcInvoker('mkvpropedit')
        tools.ccextractor = proc_invoker.MockProcInvoker('ccextractor')
        tools.subtitle_edit = proc_invoker.MockProcInvoker('subtitle-edit')

    def tearDown(self) -> None:
        tools.ffmpeg.verify()
        tools.ffprobe.verify()
        tools.comskip.verify()
        tools.comskip_gui.verify()
        tools.mkvpropedit.verify()
        tools.ccextractor.verify()
        tools.subtitle_edit.verify()

    def _read_file(self, path):
        with open(path, 'rt') as f:
            result = f.read()
        f.close()
        return result

    def _mock_ffprobe(self, input_info_basename: str):
        tools.ffprobe = proc_invoker.MockProcInvoker('ffprobe', mocks=[
            {'method_name': 'check_output', 'result': self._read_file(f'../fixtures/{input_info_basename}')},
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
                             '2 mapped stream expected: ' + str(arguments))
            original_output = list(filter(lambda e: '.original.' in e, arguments))[0]
            filtered_output = list(filter(lambda e: '.filtered.' in e, arguments))[0]
            words_output = list(filter(lambda e: '.words.' in e, arguments))[0]
            shutil.copy(f"../fixtures/{original_basename}", original_output)
            shutil.copy(f"../fixtures/{filtered_basename}", filtered_output)
            shutil.copy(f"../fixtures/{words_basename}", words_output)
            return 0

        return mock

    def _mock_ffmpeg_extract_audio_for_transcribing(self, raw_audio_basename: str):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertTrue('-map' in arguments and 's16le' in arguments, arguments)
            with open(f"../fixtures/{raw_audio_basename}", "rb") as f:
                audio = f.read()
            return audio

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

    def _mock_ffmpeg_create_with_filtered_streams(self, input_file_count: int):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertEqual(input_file_count, len(list(filter(lambda e: e == '-i', arguments))),
                             f'{input_file_count} inputs expected: ' + str(arguments))
            self.assertTrue("'-metadata:s:s:0', 'title=Filtered'" in str(arguments),
                            'Filtered subtitle expected: ' + str(arguments))
            self.assertTrue("'-metadata:s:s:1', 'title=Filtered Only'" in str(arguments),
                            'Filtered Only subtitle expected: ' + str(arguments))
            self.assertTrue("'-metadata:s:s:2', 'title=Original'" in str(arguments),
                            'Original subtitle expected: ' + str(arguments))
            self.assertTrue("'-metadata:s:s:3', 'title=Words'" in str(arguments),
                            'Words subtitle expected: ' + str(arguments))
            self.assertTrue("'-metadata:s:a:0', 'title=Filtered'" in str(arguments),
                            'Filtered audio expected: ' + str(arguments))
            self.assertTrue("'-metadata:s:a:1', 'title=Original'" in str(arguments),
                            'Original audio expected: ' + str(arguments))
            self.assertTrue("':volume=0" in str(arguments), 'audio filter with mute expected: ' + str(arguments))
            return 0

        return mock

    def _mock_ffmpeg_create_with_nofiltered_streams(self, input_file_count: int):
        def mock(method_name: str, arguments: list, **kwargs):
            self.assertEqual(input_file_count, len(list(filter(lambda e: e == '-i', arguments))),
                             f'{input_file_count} inputs expected: ' + str(arguments))
            self.assertTrue("'-metadata:s:s:0', 'title=Original'" in str(arguments),
                            'Original subtitle expected: ' + str(arguments))
            self.assertTrue("'-metadata:s:s:1', 'title=Words'" in str(arguments),
                            'Words subtitle expected: ' + str(arguments))
            self.assertFalse("'title=Filtered'" in str(arguments), 'Filtered streams not expected: ' + str(arguments))
            self.assertFalse("'title=Filtered Only'" in str(arguments),
                             'Filtered Only streams not expected: ' + str(arguments))
            self.assertFalse("':volume=0" in str(arguments), 'audio filter with mute unexpected: ' + str(arguments))
            return 0

        return mock

    def _mock_mkvpropedit_tags(self):
        def mock(method_name: str, arguments: list, **kwargs):
            return 0

        return mock

    def test_apply_unfiltered_sub_orig_text__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_filtered.raw"),
            self._mock_ffmpeg_extract_subtitle_original('needs_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_text__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_not_filtered.raw"),
            self._mock_ffmpeg_extract_subtitle_original('not_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_nofiltered_streams(4),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_image__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_image.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_dvdsub,
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_filtered.raw"),
            self._mock_ffmpeg_create_with_filtered_streams(5),
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
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_not_filtered.raw"),
            self._mock_ffmpeg_create_with_nofiltered_streams(5),
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
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_filtered.raw"),
            self._mock_ffmpeg_extract_subtitle_original('needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_image_text__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_image_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_not_filtered.raw"),
            self._mock_ffmpeg_extract_subtitle_original('not_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_nofiltered_streams(4),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_none__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_none.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_filtered.raw"),
            self._mock_ffmpeg_create_with_filtered_streams(5),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_unfiltered_sub_orig_none__notfiltered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_unfiltered_sub_orig_none.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_not_filtered.raw"),
            self._mock_ffmpeg_create_with_nofiltered_streams(5),
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
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_filtered.raw"),
            self._mock_ffmpeg_extract_subtitle_original_filtered('needs_filtered.ssa.txt', 'needs_filtered.ssa.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_apply_filtered_sub_orig_image_sub_filtered_sub_words_none__filtered(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_image_sub_filtered_sub_words_none.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_audio_for_transcribing("s16le_filtered.raw"),
            self._mock_ffmpeg_extract_subtitle_original_filtered('needs_filtered.srt.txt', 'needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(4),
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
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_extract_subtitle_original_filtered_words('needs_filtered.ssa.txt',
                                                                       'needs_filtered.ssa.txt',
                                                                       'needs_filtered.srt.txt'),
            self._mock_ffmpeg_create_with_filtered_streams(3),
        ])
        profanity_filter.do_profanity_filter(mkv_path)

    def test_remove_filtered_sub_orig_text_sub_filtered_sub_words_text(self):
        fd, mkv_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        self._mock_ffprobe('media_state_filtered_sub_orig_text_sub_filtered_sub_words_text.json')
        tools.ffmpeg = proc_invoker.MockProcInvoker('ffmpeg', mocks=[
            self._mock_ffmpeg_create_with_nofiltered_streams(1)
        ])
        profanity_filter.do_profanity_filter(mkv_path, filter_skip=True)
