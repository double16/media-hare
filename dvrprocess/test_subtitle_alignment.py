import logging
import os
import re
import tempfile
import unittest
from pathlib import Path
from typing import Union

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
        self._caplog.set_level(logging.DEBUG)

    def _read_subtitle(self, filename: str) -> Union[AssFile, SubRipFile]:
        p = Path(f"../fixtures/{filename}")
        if filename.endswith('.srt'):
            return pysrt.open(p)
        else:
            return read_ass(p)

    def _read_words_srt(self, filename: str) -> SubRipFile:
        subtitle = pysrt.open(f"../fixtures/{filename}")
        return subtitle

    def _assert_alignment(self, aligned_filename: str, original_filename: str, words_filename: str):
        expected_alignment = subtitle.open_subtitle_file_facade(Path(f"../fixtures/{aligned_filename}"))
        expected_alignment_events = list(map(lambda e: e[1], expected_alignment.events()))
        original = subtitle.open_subtitle_file_facade(Path(f"../fixtures/{original_filename}"))
        words = self._read_words_srt(words_filename)
        changed, _ = profanity_filter.fix_subtitle_audio_alignment(original.file, words)
        failed = []
        original_count = 0
        for idx, actual_event in original.events():
            original_count += 1
            if idx < len(expected_alignment_events):
                expected_event = expected_alignment_events[idx]
                expected_start_diff = abs(expected_event.start() - actual_event.start())
                if actual_event.text() != expected_event.text():
                    failed.append((
                        s_to_ts(expected_event.start() / 1000),
                        s_to_ts(actual_event.start() / 1000),
                        expected_start_diff, f"Event mismatch {idx}: '{actual_event.text()}' != '{expected_event.text()}'")
                    )
                    break
                if expected_start_diff > 400:
                    failed.append((
                        s_to_ts(expected_event.start() / 1000),
                        s_to_ts(actual_event.start() / 1000),
                        expected_start_diff, f"Start event {idx}: '{actual_event.text()}'")
                    )
                expected_end_diff = abs(expected_event.end() - actual_event.end())
                if expected_end_diff > 400:
                    failed.append((
                        s_to_ts(expected_event.end() / 1000),
                        s_to_ts(actual_event.end() / 1000),
                        expected_end_diff, f"End event {idx}: '{actual_event.text()}'")
                    )
            else:
                failed.append((
                    0,
                    s_to_ts(actual_event.start() / 1000),
                    actual_event.start(), f"Missing event {idx}: '{actual_event.text()}'")
                )
        failed.sort(key=lambda e: e[2], reverse=True)

        actual_fd, actual_path = tempfile.mkstemp(prefix=aligned_filename.split('aligned.')[0],
                                                  suffix='.actual-aligned.' + aligned_filename.split('aligned.')[1])
        os.close(actual_fd)
        original.write(Path(actual_path))
        log_path = os.path.join(os.path.dirname(actual_path), aligned_filename.split('-aligned.')[0]+'.log')
        # log_path = actual_path.replace('.ssa', '.log').replace('.srt', '.log')
        with open(log_path, 'w') as f:
            f.write(self._caplog.text)
        print(f"Wrote actual aligned file to {actual_path}, log to {log_path}")

        self.assertEqual(0, len(failed), str(failed) + "\n" + str(len(failed)) + "/" + str(original_count * 2))
        # self.assertEqual(0, len(failed), str(len(failed)) + "/" + str(len(original.events)*2))
        self.assertEqual(True, changed, 'fix_subtitle_audio_alignment should have returned changed')

    def _assert_idempotent(self, original_filename: str, words_filename: str):
        original = self._read_subtitle(original_filename)
        words = self._read_words_srt(words_filename)
        changed1, _ = profanity_filter.fix_subtitle_audio_alignment(original, words)
        changed2, _ = profanity_filter.fix_subtitle_audio_alignment(original, words)
        log_fd, log_path = tempfile.mkstemp(prefix=original_filename.split('-original.')[0]+'-aligned.', suffix='.log')
        os.write(log_fd, self._caplog.text.encode('utf-8'))
        os.close(log_fd)
        print(f'idempotent log at {log_path}')
        self.assertEqual(True, changed1, 'fix_subtitle_audio_alignment should have returned changed for first run')
        self.assertEqual(False, changed2, 'fix_subtitle_audio_alignment should have returned unchanged for second run')

    def test_bones_s02e11(self):
        self._assert_alignment('bones-s02e11-aligned.ssa', 'bones-s02e11-original.ssa', 'bones-s02e11-words.srt')

    def test_idempotent_bones_s02e11(self):
        self._assert_idempotent('bones-s02e11-original.ssa', 'bones-s02e11-words.srt')

    def test_house_s03e07(self):
        self._assert_alignment('house-s03e07-aligned.ssa', 'house-s03e07-original.ssa', 'house-s03e07-words.srt')

    def test_idempotent_house_s03e07(self):
        self._assert_idempotent('house-s03e07-original.ssa', 'house-s03e07-words.srt')

    def test_meninblack(self):
        self._assert_alignment('men-in-black-aligned.srt', 'men-in-black-original.srt', 'men-in-black-words.srt')

    def test_idempotent_meninblack(self):
        self._assert_idempotent('men-in-black-original.srt', 'men-in-black-words.srt')

    def test_subtitle_text_to_plain(self):
        self.assertEqual(
            ['worked there for fifteen years'],
            profanity_filter._subtitle_text_to_plain('Worked there for 15 years.'),
        )
        self.assertEqual(
            ['it costs twenty three dollar zero cents'],
            profanity_filter._subtitle_text_to_plain('It costs $23'),
        )
        self.assertEqual(
            ['call eight hundred five fifty five one thousand two hundred and twelve',
             'call eight hundred five fifty five one two one two',
             'call eight hundred five fifty five twelve twelve',
             'call eight hundred five five five one thousand two hundred and twelve',
             'call eight hundred five five five one two one two',
             'call eight hundred five five five twelve twelve',
             'call eight hundred five hundred and fifty five one thousand two hundred and '
             'twelve',
             'call eight hundred five hundred and fifty five one two one two',
             'call eight hundred five hundred and fifty five twelve twelve',
             'call eight zero five fifty five one thousand two hundred and twelve',
             'call eight zero five fifty five one two one two',
             'call eight zero five fifty five twelve twelve',
             'call eight zero five five five one thousand two hundred and twelve',
             'call eight zero five five five one two one two',
             'call eight zero five five five twelve twelve',
             'call eight zero five hundred and fifty five one thousand two hundred and '
             'twelve',
             'call eight zero five hundred and fifty five one two one two',
             'call eight zero five hundred and fifty five twelve twelve',
             'call eight zero zero five fifty five one thousand two hundred and twelve',
             'call eight zero zero five fifty five one two one two',
             'call eight zero zero five fifty five twelve twelve',
             'call eight zero zero five five five one thousand two hundred and twelve',
             'call eight zero zero five five five one two one two',
             'call eight zero zero five five five twelve twelve',
             'call eight zero zero five hundred and fifty five one thousand two hundred '
             'and twelve',
             'call eight zero zero five hundred and fifty five one two one two',
             'call eight zero zero five hundred and fifty five twelve twelve'],
            profanity_filter._subtitle_text_to_plain('Call (800) 555-1212'),
        )
