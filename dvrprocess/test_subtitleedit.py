import logging
import os
import shutil
import tempfile
import unittest

from common import tools, constants
from profanity_filter import create_subtitle_idx_file, words_in_dictionary_pct, WORD_FOUND_PCT_THRESHOLD


class SubtitleEditTest(unittest.TestCase):
    subtitle_edit_present = None

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    @classmethod
    def setUpClass(cls):
        cls.subtitle_edit_present = tools.subtitle_edit.present()

    def _check_tools(self):
        if not self.subtitle_edit_present:
            self.skipTest('subtitle-edit not available')

    def _create_idx_file(self, vob_filename: str):
        idx_filename = vob_filename.removesuffix('.vob') + '.idx'
        create_subtitle_idx_file(idx_filename, 720, 480)

    def _assert_word_pct_threshold(self, fixture_filename: str):
        self._check_tools()
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.vob') as dest:
            dest.file.close()
            source_subtitle_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../fixtures/{fixture_filename}'))
            subtitle_filename = dest.name
            shutil.copy(source_subtitle_filename, subtitle_filename)
            self.assertTrue(os.stat(subtitle_filename).st_size > 0, 'copy to temp file failed')
            if subtitle_filename.endswith('.vob'):
                srt_filename = subtitle_filename.removesuffix('.vob') + '.srt'
                self._create_idx_file(subtitle_filename)
            else:
                srt_filename = subtitle_filename.removesuffix('.sub') + '.srt'
            tools.subtitle_edit.run([subtitle_filename], check=True)

            self.assertTrue(os.access(srt_filename, os.R_OK), "SRT not generated from OCR")

            word_found_pct = words_in_dictionary_pct(srt_filename, constants.LANGUAGE_ENGLISH, 120)
            self.assertTrue(word_found_pct >= WORD_FOUND_PCT_THRESHOLD - 10,
                            f"OCR text appears to be incorrect, {word_found_pct}% words found in English dictionary")

    def test_dvd_subtitle1(self):
        self._assert_word_pct_threshold('dvd_subtitle1.vob')

    def test_dvd_subtitle2(self):
        self._assert_word_pct_threshold('dvd_subtitle2.vob')

    def test_dvd_subtitle3(self):
        self._assert_word_pct_threshold('dvd_subtitle3.vob')

    def test_dvd_subtitle4(self):
        self._assert_word_pct_threshold('dvd_subtitle4.vob')

    def test_bluray_subtitle1(self):
        self._assert_word_pct_threshold('bluray_subtitle1.sup')

    def test_bluray_subtitle2(self):
        self._assert_word_pct_threshold('bluray_subtitle2.sup')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
