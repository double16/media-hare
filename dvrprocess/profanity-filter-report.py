#!/usr/bin/env python3

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pysrt
from ass_parser import read_ass, errors

import common

logger = logging.getLogger(__name__)


def profanity_filter_report_cli(argv):
    ffmpeg = common.find_ffmpeg()
    data = []
    for file, _ in common.generate_video_files(argv):
        pf_data = extract_pf_data(file, ffmpeg)
        if pf_data:
            data.append(pf_data)
    print(json.dumps(data, indent=2))


def extract_pf_data(mkv, ffmpeg):
    input_info = common.find_input_info(mkv)
    subtitle_original, subtitle_filtered, _ = common.find_original_and_filtered_streams(input_info,
                                                                                        common.CODEC_SUBTITLE,
                                                                                        [
                                                                                            common.CODEC_SUBTITLE_ASS,
                                                                                            common.CODEC_SUBTITLE_SRT,
                                                                                            common.CODEC_SUBTITLE_SUBRIP],
                                                                                        common.LANGUAGE_ENGLISH)
    if subtitle_original is None or subtitle_filtered is None:
        print(f"INFO: {mkv} has no filtered subtitles", file=sys.stderr)
        return None

    data = {
        'filename': mkv,
        common.K_FILTER_VERSION: input_info[common.K_FORMAT][common.K_TAGS][common.K_FILTER_VERSION],
        common.K_FILTER_HASH: input_info[common.K_FORMAT][common.K_TAGS][common.K_FILTER_HASH],
        common.K_FILTER_STOPPED: input_info[common.K_FORMAT][common.K_TAGS].get(common.K_FILTER_STOPPED, None),
        common.K_AUDIO_TO_TEXT_VERSION: input_info[common.K_FORMAT][common.K_TAGS].get(common.K_AUDIO_TO_TEXT_VERSION,
                                                                                       None),
        'changes': []
    }
    subtitle_codec = subtitle_original['codec_name']
    if subtitle_codec == common.CODEC_SUBTITLE_SUBRIP:
        subtitle_codec = common.CODEC_SUBTITLE_SRT
    fd, file_original = tempfile.mkstemp(suffix='.' + subtitle_codec, prefix='.~')
    os.close(fd)
    fd, file_filtered = tempfile.mkstemp(suffix='.' + subtitle_codec, prefix='.~')
    os.close(fd)
    extract_command = [ffmpeg, '-hide_banner', '-y', '-analyzeduration', common.ANALYZE_DURATION,
                       '-probesize', common.PROBE_SIZE,
                       '-i', input_info['format']['filename'],
                       '-c:s', 'copy',
                       '-map', f'0:{subtitle_original[common.K_STREAM_INDEX]}', file_original,
                       '-map', f'0:{subtitle_filtered[common.K_STREAM_INDEX]}', file_filtered
                       ]
    subprocess.run(extract_command, check=True, capture_output=True)

    print("INFO: %s, %s, original %s bytes, filtered %s bytes",
          mkv, subtitle_codec, os.stat(file_original).st_size, os.stat(file_filtered).st_size,
          file=sys.stderr)

    parsed_original = None
    parsed_filtered = None
    try:
        if subtitle_codec == common.CODEC_SUBTITLE_ASS:
            parsed_original = list(
                map(lambda e: {'when': e.start, 'text': e.text}, read_ass(Path(file_original)).events))
            parsed_filtered = list(
                map(lambda e: {'when': e.start, 'text': e.text}, read_ass(Path(file_filtered)).events))
        elif subtitle_codec in [common.CODEC_SUBTITLE_SRT, common.CODEC_SUBTITLE_SUBRIP]:
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
    for i, val in enumerate(parsed_filtered):
        if '***' in val['text']:
            data['changes'].append({
                'when': common.s_to_ts(parsed_original[i]['when'] / 1000.0),
                'original': parsed_original[i]['text'],
                'filtered': val['text']})

    return data


if __name__ == '__main__':
    common.setup_cli()
    profanity_filter_report_cli(sys.argv[1:])
