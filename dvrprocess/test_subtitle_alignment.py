import os
import re
import tempfile
import unittest
from pathlib import Path

import pysrt
from pysrt import SubRipFile
from ass_parser import read_ass, AssFile
from common import s_to_ts

import profanity_filter


class SubtitleAlignmentTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.event_re = re.compile(r'Dialogue: \d,')
        self.alignment_re = re.compile(r',\d\d:\d\d:\d\d[.]\d\d\d,\d\d:\d\d:\d\d[.]\d\d\d,')

    def _read_aligned_and_original_ass(self, aligned_filename: str, original_filename: str) -> (AssFile, AssFile):
        """
        Read the expected alignment. This is special because only event with start and end that have an SRT like
        timestamp will be used.
        :return: aligned, original
        """
        aligned_fd, filtered_aligned_path = tempfile.mkstemp(suffix='.ssa')
        original_fd, filtered_original_path = tempfile.mkstemp(suffix='.ssa')
        with open(f"../fixtures/{aligned_filename}", "rt") as aligned_source:
            with open(f"../fixtures/{original_filename}", "rt") as original_source:
                original_lines = original_source.readlines()
                for idx, line in enumerate(aligned_source.readlines()):
                    if not self.event_re.search(line) or self.alignment_re.search(line):
                        os.write(aligned_fd, bytes(line, 'utf8'))
                        os.write(original_fd, bytes(original_lines[idx], 'utf8'))
        os.close(aligned_fd)
        os.close(original_fd)
        aligned_subtitle = read_ass(Path(filtered_aligned_path))
        original_subtitle = read_ass(Path(filtered_original_path))
        return aligned_subtitle, original_subtitle

    def _read_words_srt(self, filename: str) -> SubRipFile:
        subtitle = pysrt.open(f"../fixtures/{filename}")
        return subtitle

    def _assert_alignment(self, aligned_filename: str, original_filename: str, words_filename: str):
        expected_alignment, original = self._read_aligned_and_original_ass(aligned_filename,
                                                                           original_filename)
        words = self._read_words_srt(words_filename)
        changed = profanity_filter.fix_subtitle_audio_alignment(original, words, original_filename)
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
        changed1 = profanity_filter.fix_subtitle_audio_alignment(original, words, 'bones-s02e11-original.ssa')
        changed2 = profanity_filter.fix_subtitle_audio_alignment(original, words, 'bones-s02e11-original.ssa')
        self.assertEqual(True, changed1, 'fix_subtitle_audio_alignment should have returned changed for first run')
        self.assertEqual(False, changed2, 'fix_subtitle_audio_alignment should have returned unchanged for second run')

    def test_house_s03e07(self):
        self._assert_alignment('house-s03e07-aligned.ssa', 'house-s03e07-original.ssa', 'house-s03e07-words.srt')
