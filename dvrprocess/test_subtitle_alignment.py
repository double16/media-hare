import logging
import os
import re
import tempfile
import unittest
from pathlib import Path

import pysrt
import pytest
from pysrt import SubRipFile
from ass_parser import read_ass, write_ass, AssFile
from common import s_to_ts, subtitle

import profanity_filter


class SubtitleAlignmentTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.event_re = re.compile(r'Dialogue: \d,')
        self.alignment_re = re.compile(r',\d\d:\d\d:\d\d[.]\d\d\d,\d\d:\d\d:\d\d[.]\d\d\d,')

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def _read_aligned_and_original_ass(self, aligned_filename: str, original_filename: str) -> (AssFile, AssFile):
        """
        Read the expected alignment. This is special because only event with start and end that have an SRT like
        timestamp will be used.
        :return: aligned, original
        """
        suffix = original_filename.split('.')[-1]
        aligned_fd, filtered_aligned_path = tempfile.mkstemp(suffix=suffix)
        os.close(aligned_fd)
        original_fd, filtered_original_path = tempfile.mkstemp(suffix=suffix)
        os.close(original_fd)

        aligned = subtitle.open_subtitle_file_facade(Path(aligned_filename))
        aligned_dict = dict()
        for event_idx, event in aligned.events():
            aligned_dict[event.text().__hash__()] = event
        aligned.write(Path(filtered_aligned_path))

        original = subtitle.open_subtitle_file_facade(Path(original_filename))
        original_dict = dict()
        for event_idx, event in reversed(list(original.events())):
            if event.text().__hash__() not in aligned_dict:

                original_dict[event.text().__hash__()] = event
        original.write(Path(filtered_original_path))

        aligned_subtitle = read_ass(Path(filtered_aligned_path))
        original_subtitle = read_ass(Path(filtered_original_path))
        os.remove(filtered_aligned_path)
        os.remove(filtered_original_path)
        return aligned, original_subtitle

    def _read_words_srt(self, filename: str) -> SubRipFile:
        subtitle = pysrt.open(f"../fixtures/{filename}")
        return subtitle

    def _assert_alignment(self, aligned_filename: str, original_filename: str, words_filename: str):
        expected_alignment, original = self._read_aligned_and_original_ass(aligned_filename,
                                                                           original_filename)
        words = self._read_words_srt(words_filename)
        changed = profanity_filter.fix_subtitle_audio_alignment(original, words)
        failed = []
        for idx, actual_event in enumerate(original.events):
            expected_event = expected_alignment.events[idx]
            expected_start_diff = abs(expected_event.start - actual_event.start)
            if expected_start_diff > 400:
                failed.append((
                    s_to_ts(expected_event.start/1000),
                    s_to_ts(actual_event.start/1000),
                    expected_start_diff, f"Start event {idx}: '{actual_event.text}'")
                )
            expected_end_diff = abs(expected_event.end - actual_event.end)
            if expected_end_diff > 400:
                failed.append((
                    s_to_ts(expected_event.end/1000),
                    s_to_ts(actual_event.end/1000),
                    expected_end_diff, f"End event {idx}: '{actual_event.text}'")
                )
        failed.sort(key=lambda e: e[2], reverse=True)
        self.assertEqual(0, len(failed), str(failed) + "\n" + str(len(failed)) + "/" + str(len(original.events)*2))
        # self.assertEqual(0, len(failed), str(len(failed)) + "/" + str(len(original.events)*2))
        self.assertEqual(True, changed, 'fix_subtitle_audio_alignment should have returned changed')

    def test_bones_s02e11(self):
        self._assert_alignment('bones-s02e11-aligned.ssa', 'bones-s02e11-original.ssa', 'bones-s02e11-words.srt')

    def test_idempotent(self):
        expected_alignment, original = self._read_aligned_and_original_ass('bones-s02e11-aligned.ssa', 'bones-s02e11-original.ssa')
        words = self._read_words_srt('bones-s02e11-words.srt')
        changed1 = profanity_filter.fix_subtitle_audio_alignment(original, words)
        changed2 = profanity_filter.fix_subtitle_audio_alignment(original, words)
        self.assertEqual(True, changed1, 'fix_subtitle_audio_alignment should have returned changed for first run')
        self.assertEqual(False, changed2, 'fix_subtitle_audio_alignment should have returned unchanged for second run')

    def test_house_s03e07(self):
        self._assert_alignment('house-s03e07-aligned.ssa', 'house-s03e07-original.ssa', 'house-s03e07-words.srt')

    def test_meninblack(self):
        self._caplog.set_level(logging.DEBUG)
        self._assert_alignment('men-in-black-aligned.srt', 'men-in-black-original.srt', 'men-in-black-words.srt')

    def test_house_s03e07_output(self):
        original = read_ass(Path('../fixtures/house-s03e07-original.ssa'))
        words = self._read_words_srt('house-s03e07-words.srt')
        profanity_filter.fix_subtitle_audio_alignment(original, words)
        aligned_fd, aligned_path = tempfile.mkstemp(prefix='house-s03e07-aligned', suffix='.ssa')
        os.close(aligned_fd)
        write_ass(original, Path(aligned_path))
        print(f'house-s03e07 alignment in {aligned_path}')

    def test_bones_s02e11_output(self):
        original = read_ass(Path('../fixtures/bones-s02e11-original.ssa'))
        words = self._read_words_srt('bones-s02e11-words.srt')
        profanity_filter.fix_subtitle_audio_alignment(original, words)
        aligned_fd, aligned_path = tempfile.mkstemp(prefix='bones-s02e11-aligned', suffix='.ssa')
        os.close(aligned_fd)
        write_ass(original, Path(aligned_path))
        print(f'bones-s02e11 alignment in {aligned_path}')

    def test_meninblack_output(self):
        self._caplog.set_level(logging.DEBUG)
        original = pysrt.open(Path('../fixtures/men-in-black-original.srt'))
        words = self._read_words_srt('men-in-black-words.srt')
        profanity_filter.fix_subtitle_audio_alignment(original, words)
        aligned_fd, aligned_path = tempfile.mkstemp(prefix='men-in-black-aligned', suffix='.srt')
        os.close(aligned_fd)
        original.save(Path(aligned_path), 'utf-8')
        with open(aligned_path.replace('.srt', '.log'), 'w') as f:
            f.write(self._caplog.text)
        print(f'men-in-black alignment in {aligned_path}')
