#!/usr/bin/env python3

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pysrt
from ass_parser import read_ass, errors

import common
from common import tools, constants, subtitle

logger = logging.getLogger(__name__)


def profanity_filter_report_cli(argv):
    data = []
    for file, _ in common.generate_video_files(argv, fail_on_missing=True):
        pf_data = extract_pf_data(file)
        if pf_data:
            data.append(pf_data)
    print(json.dumps(data, indent=2))


def extract_pf_data(mkv):
    input_info = common.find_input_info(mkv)
    subtitle_original, subtitle_filtered, _, _ = common.find_original_and_filtered_streams(input_info,
                                                                                           constants.CODEC_SUBTITLE,
                                                                                           [
                                                                                               constants.CODEC_SUBTITLE_ASS,
                                                                                               constants.CODEC_SUBTITLE_SRT,
                                                                                               constants.CODEC_SUBTITLE_SUBRIP],
                                                                                           constants.LANGUAGE_ENGLISH)
    if subtitle_original is None or subtitle_filtered is None:
        print(f"INFO: {mkv} has no filtered subtitles", file=sys.stderr)
        return None

    tags = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {})
    data = {
        'filename': mkv,
        constants.K_FILTER_VERSION: tags.get(constants.K_FILTER_VERSION, None),
        constants.K_FILTER_HASH: tags.get(constants.K_FILTER_HASH, None),
        constants.K_FILTER_STOPPED: tags.get(constants.K_FILTER_STOPPED, None),
        constants.K_AUDIO_TO_TEXT_VERSION: tags.get(constants.K_AUDIO_TO_TEXT_VERSION, None),
        'changes': []
    }
    subtitle_codec = subtitle_original['codec_name']
    if subtitle_codec == constants.CODEC_SUBTITLE_SUBRIP:
        subtitle_codec = constants.CODEC_SUBTITLE_SRT
    fd, file_original = tempfile.mkstemp(suffix='.' + subtitle_codec, prefix='.~')
    os.close(fd)
    fd, file_filtered = tempfile.mkstemp(suffix='.' + subtitle_codec, prefix='.~')
    os.close(fd)
    extract_command = ['-hide_banner', '-y', '-analyzeduration', common.ANALYZE_DURATION,
                       '-probesize', common.PROBE_SIZE,
                       '-i', input_info['format']['filename'],
                       '-c:s', 'copy',
                       '-map', f'0:{subtitle_original[constants.K_STREAM_INDEX]}', file_original,
                       '-map', f'0:{subtitle_filtered[constants.K_STREAM_INDEX]}', file_filtered
                       ]
    tools.ffmpeg.run(extract_command, check=True, capture_output=True)

    print(
        f"INFO: {mkv}, {subtitle_codec}, original {os.stat(file_original).st_size} bytes, filtered {os.stat(file_filtered).st_size} bytes",
        file=sys.stderr)

    parsed_original = None
    parsed_filtered = None
    try:
        if subtitle_codec == constants.CODEC_SUBTITLE_ASS:
            parsed_original = list(
                map(lambda e: {'when': e.start, 'text': e.text},
                    read_ass(subtitle.clean_ssa(Path(file_original))).events))
            parsed_filtered = list(
                map(lambda e: {'when': e.start, 'text': e.text},
                    read_ass(subtitle.clean_ssa(Path(file_filtered))).events))
        elif subtitle_codec in [constants.CODEC_SUBTITLE_SRT, constants.CODEC_SUBTITLE_SUBRIP]:
            parsed_original = list(map(lambda e: {'when': e.start.ordinal, 'text': e.text}, pysrt.open(file_original)))
            parsed_filtered = list(map(lambda e: {'when': e.start.ordinal, 'text': e.text}, pysrt.open(file_filtered)))
        else:
            common.fatal(f"INFO: Unknown subtitle codec {subtitle_codec}")
    except errors.CorruptAssError as e:
        logger.error("%s: %s", mkv, str(e))
        return None
    finally:
        if os.path.exists(file_original):
            os.remove(file_original)
        if os.path.exists(file_filtered):
            os.remove(file_filtered)

    print(f"INFO: {mkv} {len(parsed_filtered)} subtitle events", file=sys.stderr)
    # FIXME: this break with subtitle alignment / new events
    for i, val in enumerate(parsed_filtered):
        if '***' in val['text']:
            data['changes'].append({
                'when': common.s_to_ts(parsed_original[i]['when'] / 1000.0),
                'original': parsed_original[i]['text'],
                'filtered': val['text']})

    return data


if __name__ == '__main__':
    common.setup_cli(level=logging.ERROR, start_gauges=False)
    profanity_filter_report_cli(sys.argv[1:])
