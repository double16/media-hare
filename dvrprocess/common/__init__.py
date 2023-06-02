import _thread
import code
import datetime
import json
import logging
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
from enum import Enum
from typing import Union
from xml.etree import ElementTree as ET

import psutil
from psutil import AccessDenied, NoSuchProcess

from . import hwaccel, tools, config, constants

_allocate_lock = _thread.allocate_lock
_once_lock = _allocate_lock()

logger = logging.getLogger(__name__)

# Temporary File Name for transcoding, we want to keep on the same filesystem as the input
TEMPFILENAME = None
TEMPFILENAMES = []
# We need this because shell escaping is hard for some ffmpeg options, specifically the lavfi filter
FILENAME_CLEAN_LINK = None
# Hides filename from user UI and Dropbox and used to de-conflict when source and target name are equal
HIDDEN_FILENAME = None

ANALYZE_DURATION = '20000000'
PROBE_SIZE = '20000000'

MEDIA_BASE = None


def get_media_base():
    global MEDIA_BASE
    if MEDIA_BASE is None:
        _once_lock.acquire()
        try:
            if MEDIA_BASE is None:
                MEDIA_BASE = find_media_base()
        finally:
            _once_lock.release()
    return MEDIA_BASE


def find_media_base():
    paths = config.get_global_config_option('media', 'paths').split(',')
    if len(paths) > 0:
        for p in psutil.disk_partitions(all=True):
            if os.path.isdir(os.path.join(p.mountpoint, paths[0])):
                return p.mountpoint
    for root in config.get_global_config_option('media', 'root').split(','):
        root = root.replace('$HOME', os.environ['HOME'])
        if os.path.isdir(root):
            return root
    raise FileNotFoundError('No media.root in config')


def get_media_paths(base=None):
    if base is None:
        base = get_media_base()
    paths = config.get_global_config_option('media', 'paths').split(',')
    return list(map(lambda e: os.path.join(base, e), paths))


def fatal(message):
    logger.fatal(message)
    sys.exit(255)


def exception_hook(exctype, value, traceback_obj):
    logger.fatal("Uncaught exception", exc_info=(exctype, value, traceback_obj))
    sys.exit(255)


sys.excepthook = exception_hook


def finish():
    global TEMPFILENAME
    global FILENAME_CLEAN_LINK
    if TEMPFILENAME and os.path.isfile(TEMPFILENAME):
        try:
            os.remove(TEMPFILENAME)
        except FileNotFoundError:
            pass
    if FILENAME_CLEAN_LINK and os.path.islink(FILENAME_CLEAN_LINK):
        try:
            os.remove(FILENAME_CLEAN_LINK)
        except FileNotFoundError:
            pass
    for FN in TEMPFILENAMES:
        if os.path.isfile(FN):
            try:
                os.remove(FN)
            except FileNotFoundError:
                pass
    TEMPFILENAMES.clear()


def finisher(func):
    """
    Calls finish() after execution.
    """

    def inner1(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            finish()

    return inner1


def get_plex_url():
    return config.get_global_config_option('plex', 'url', fallback=None)


def load_keyframes_by_seconds(filepath) -> list[float]:
    ffprobe_keyframes = ["-loglevel", "error", "-skip_frame", "nointra", "-select_streams", "v:0",
                         "-show_entries",
                         "frame=pkt_pts_time" if int(tools.ffprobe.version) == 4 else "frame=pts_time",
                         "-of", "csv=print_section=0", filepath]
    keyframes = list(map(lambda s: float(re.search(r'[\d.]+', s)[0]),
                         filter(lambda s: len(s) > 0,
                                tools.ffprobe.check_output(ffprobe_keyframes, universal_newlines=True,
                                                           text=True).splitlines())))
    if len(keyframes) == 0:
        raise ChildProcessError("No key frames returned, suspect ffprobe command line is broken")

    return keyframes


class KeyframeSearchPreference(Enum):
    CLOSEST = 0
    AFTER = 1
    BEFORE = 2


def find_desired_keyframe(keyframes: list[float], target_time: float,
                          search_preference: KeyframeSearchPreference = KeyframeSearchPreference.CLOSEST,
                          start_time: [None, float] = None) -> float:
    """
    Find the desired keyframe for the target_time.
    :param keyframes: list of keyframes in the video with times directly from the video
    :param target_time: time we want, zero based, because it's a human-readable time
    :param search_preference:
    :param start_time:
    :return:
    """
    if start_time is None:
        if len(keyframes) > 0:
            start_time = keyframes[0]
        else:
            start_time = 0.0

    if len(keyframes) == 0:
        return target_time + start_time

    return _find_desired_keyframe(keyframes, target_time, start_time, search_preference)


def _find_desired_keyframe(keyframes: list[float], target_time: float, start_time: float,
                           search_preference: KeyframeSearchPreference = KeyframeSearchPreference.CLOSEST,
                           ) -> float:
    if len(keyframes) == 0:
        return target_time + start_time
    elif len(keyframes) == 1:
        return keyframes[0]
    elif len(keyframes) == 2:
        if search_preference == KeyframeSearchPreference.AFTER:
            if _keyframe_compare(keyframes[0], target_time, start_time) >= 0:
                return keyframes[0]
            return keyframes[1]
        elif search_preference == KeyframeSearchPreference.BEFORE:
            if _keyframe_compare(keyframes[1], target_time, start_time) <= 0:
                return keyframes[1]
            return keyframes[0]

        d1 = abs(keyframes[0] - target_time)
        d2 = abs(keyframes[1] - target_time)
        if d1 <= d2:
            return keyframes[0]
        else:
            return keyframes[1]

    mid = len(keyframes) // 2
    c = _keyframe_compare(keyframes[mid], target_time, start_time)
    if c == 0:
        return keyframes[mid]
    if c > 0:
        return _find_desired_keyframe(keyframes[:mid + 1], target_time, start_time, search_preference)
    else:
        return _find_desired_keyframe(keyframes[mid:], target_time, start_time, search_preference)


def _keyframe_compare(keyframe: float, operand: float, start_time: float, tolerance: float = 0.002) -> int:
    """
    Compare a raw key frame and a zero based time.
    :param keyframe:
    :param operand:
    :param start_time:
    :param tolerance:
    :return: 0 if times are equal, -1 if keyframe < operand, 1 if keyframe > operand
    """
    keyframe_adjusted = keyframe - start_time
    if abs(keyframe_adjusted - operand) <= tolerance:
        return 0
    if keyframe_adjusted < operand:
        return -1
    if keyframe_adjusted > operand:
        return 1
    return 0


def fix_closed_caption_report(input_info, filename):
    video_info = find_video_stream(input_info)
    if video_info and 'Closed Captions' in tools.ffprobe.check_output(
            ['-analyzeduration', ANALYZE_DURATION, '-probesize', PROBE_SIZE, filename],
            text=True, stderr=subprocess.STDOUT):
        video_info['closed_captions'] = 1


def find_input_info(filename):
    input_info = json.loads(tools.ffprobe.check_output(
        ['-v', 'quiet', '-analyzeduration', ANALYZE_DURATION, '-probesize', PROBE_SIZE, '-print_format',
         'json',
         '-show_format', '-show_streams', '-show_chapters', filename]))
    fix_closed_caption_report(input_info, filename)
    return input_info


def get_video_height(video_info) -> [None, int]:
    """
    Get the height of the video considering symbolic names like 'sd'.
    :param video_info: can be all of the input_info to use default video stream, or a single video stream
    :return: int of height: 480, 720, etc. or None
    """
    if 'streams' in video_info:
        video_info = find_video_stream(video_info)
    height = video_info['height']
    if not height:
        return None
    if height == 'sd':
        height = 480
    return height


def get_video_width(video_info) -> [None, int]:
    """
    Get the width of the video considering symbolic names.
    :param video_info: can be all of the input_info to use default video stream, or a single video stream
    :return: int of width or None
    """
    if 'streams' in video_info:
        video_info = find_video_stream(video_info)
    width = video_info['width']
    if not width:
        return None
    return width


def get_frame_rate(video_info):
    """
    Get the frame rate of the video.
    :param video_info: can be all of the input_info to use default video stream, or a single video stream
    :return: str, float or None
    """
    if 'streams' in video_info:
        video_info = find_video_stream(video_info)
    frame_rate = video_info['avg_frame_rate']
    if not frame_rate:
        return None
    return frame_rate


def find_original_and_filtered_streams(input_info, codec_type, codec_names=None, language=None) -> (str, str, str, str):
    """
    Find streams produced by the profanity filter.
    :param input_info:
    :param codec_type:
    :param codec_names:
    :param language:
    :return: subtitle_original, subtitle_filtered, subtitle_filtered_forced, subtitle_words
    """
    original = None
    filtered = None
    filtered_forced = None
    words = None

    streams = find_streams_by_codec_and_language(input_info, codec_type, codec_names, language)

    filtered_streams = list(
        filter(lambda stream: stream['tags'].get(constants.K_STREAM_TITLE) == constants.TITLE_FILTERED, streams))
    if (len(filtered_streams)) > 0:
        filtered = filtered_streams[0]

    filtered_forced_streams = list(
        filter(lambda stream: stream['tags'].get(constants.K_STREAM_TITLE) == constants.TITLE_FILTERED_FORCED, streams))
    if (len(filtered_forced_streams)) > 0:
        filtered_forced = filtered_forced_streams[0]

    words_streams = list(
        filter(lambda stream: stream['tags'].get(constants.K_STREAM_TITLE) == constants.TITLE_WORDS, streams))
    if (len(words_streams)) > 0:
        words = words_streams[0]

    original_streams = list(
        filter(lambda stream: stream['tags'].get(constants.K_STREAM_TITLE) == constants.TITLE_ORIGINAL, streams))
    if (len(original_streams)) > 0:
        original = original_streams[0]
    else:
        original_streams = list(
            filter(lambda stream: stream['tags'].get(constants.K_STREAM_TITLE) != constants.TITLE_FILTERED, streams))
        if (len(original_streams)) > 0:
            original = original_streams[0]

    return original, filtered, filtered_forced, words


def find_streams_by_codec_and_language(input_info, codec_type, codec_names=None, language=None):
    # select by preferred language
    streams = list(filter(lambda stream: stream['codec_type'] == codec_type
                                         and (codec_names is None or stream['codec_name'] in codec_names)
                                         and stream['tags'].get('language') == language,
                          input_info['streams']))
    # select by unspecified language
    if len(streams) == 0:
        streams = list(filter(lambda stream: stream['codec_type'] == codec_type
                                             and (codec_names is None or stream['codec_name'] in codec_names)
                                             and stream['tags'].get('language') is None,
                              input_info['streams']))
    # select anything
    if len(streams) == 0:
        streams = list(filter(lambda stream: stream['codec_type'] == codec_type
                              and (codec_names is None or stream['codec_name'] in codec_names),
                              input_info['streams']))
    return streams


def assert_not_transcoding(input_file, tempfilename=None, exit=True):
    global TEMPFILENAME, FILENAME_CLEAN_LINK, TEMPFILENAMES
    if tempfilename is None:
        tempfilename = TEMPFILENAME
    if tempfilename is None:
        dir = os.path.dirname(os.path.abspath(input_file))
        base = os.path.basename(input_file)
        parts = base.split('.')
        tempfilename = os.path.join(dir, '.~' + '.'.join(parts[0:-1]) + '.transcoded.' + parts[-1])
    if os.path.isfile(tempfilename) and (time.time() - os.path.getmtime(tempfilename)) < 172800:
        logger.info(f"Already transcoding, skipping {input_file} ({tempfilename})")
        # We don't want clean up to remove these files and mess up other processes
        TEMPFILENAME = None
        FILENAME_CLEAN_LINK = None
        TEMPFILENAMES.clear()
        if exit:
            sys.exit(0)
        else:
            return 255
    return 0


# Return the list of English streams, or empty if none.
def find_english_streams(streams):
    try:
        return list(filter(lambda stream: stream['tags']['language'] == 'eng', streams))
    except KeyError:
        return []


def has_stream_with_language(input_info, codec_type, codec_names=None, language=None):
    return len(
        list(filter(lambda stream: stream['codec_type'] == codec_type
                                   and (codec_names is None or stream['codec_name'] in codec_names)
                                   and stream['tags'].get('language') == language,
                    input_info['streams']))) > 0


def is_video_stream(stream_info: dict) -> bool:
    return stream_info[constants.K_CODEC_TYPE] == constants.CODEC_VIDEO and (
            not stream_info.get(constants.K_DISPOSITION) or stream_info.get(constants.K_DISPOSITION).get(
                'attached_pic') != 1) and stream_info.get('avg_frame_rate') != '0/0'


def is_audio_stream(stream_info: dict) -> bool:
    return stream_info[constants.K_CODEC_TYPE] == constants.CODEC_AUDIO


def is_subtitle_text_stream(stream_info: dict) -> bool:
    if stream_info is None:
        return False
    return stream_info.get(constants.K_CODEC_TYPE, '') == constants.CODEC_SUBTITLE and stream_info.get(
        constants.K_CODEC_NAME,
        '') in constants.CODEC_SUBTITLE_TEXT_BASED


def find_video_streams(input_info) -> list[dict]:
    """
    Find all video streams that do not have other purposes, such as attached pictures.
    :param input_info:
    :return: list of stream info maps
    """
    streams = list(filter(lambda stream: is_video_stream(stream), input_info[constants.K_STREAMS]))
    return streams


def find_video_stream(input_info) -> [None, dict]:
    """
    Find the primary video stream.
    :param input_info:
    :return: stream map or None
    """
    streams = find_video_streams(input_info)
    english = find_english_streams(streams)
    if len(english) > 0:
        streams = english
    if len(streams) == 0:
        return None
    return streams[0]


def find_audio_streams(input_info):
    streams = list(filter(lambda stream: is_audio_stream(stream), input_info[constants.K_STREAMS]))

    # Check for profanity filter streams
    filter_streams = list(
        filter(lambda s: s.get('tags', {}).get(constants.K_STREAM_TITLE) in [constants.TITLE_ORIGINAL,
                                                                             constants.TITLE_FILTERED], streams))
    if len(filter_streams) == 2:
        return filter_streams

    # prefer English streams
    english = find_english_streams(streams)
    if len(english) > 0:
        streams = english
    if len(streams) < 2:
        return streams
    # Pick the largest bitrate
    if len(list(filter(lambda stream: constants.K_BIT_RATE in stream, streams))) == len(streams):
        # In case all of the bit rates are the same, we don't want to lose the order when we leave this block
        streams2 = streams.copy()
        streams2.sort(key=lambda e: e[constants.K_BIT_RATE], reverse=True)
        if streams2[0][constants.K_BIT_RATE] > streams2[1][constants.K_BIT_RATE]:
            return streams2[0:1]
    # Pick the most channels
    if len(list(filter(lambda stream: constants.K_CHANNELS in stream, streams))) == len(streams):
        # In case all of the channel counts are the same, we don't want to lose the order when we leave this block
        streams2 = streams.copy()
        streams2.sort(key=lambda e: e[constants.K_CHANNELS], reverse=True)
        if streams2[0][constants.K_CHANNELS] > streams2[1][constants.K_CHANNELS]:
            return streams2[0:1]
    # Pick default disposition
    default_streams = list(
        filter(lambda stream: stream.get(constants.K_DISPOSITION) and stream.get(constants.K_DISPOSITION).get(
            'default') > 0, streams))
    if len(default_streams) > 0:
        return default_streams[0:1]

    return streams


def find_attached_pic_stream(input_info):
    return list(filter(lambda stream: (stream.get(constants.K_DISPOSITION) and stream.get(constants.K_DISPOSITION).get(
        'attached_pic') > 0)
                                      or (stream.get(constants.K_TAGS) and stream.get(constants.K_TAGS).get('MIMETYPE',
                                                                                                            '').startswith(
        'image/')),
                       input_info[constants.K_STREAMS]))


def sort_streams(streams: list[dict]) -> list[dict]:
    """
    Sort streams in order of video, audio, other. In case of using complex filter graphs, the outputs will be ordered
    first regardless of stream mapping.
    :param streams:
    :return: sorted streams, the input is not modified
    """
    result = streams.copy()

    def stream_sort_key(stream: dict):
        codec_type = stream[constants.K_CODEC_TYPE]
        if codec_type == constants.CODEC_VIDEO:
            if stream.get(constants.K_DISPOSITION, {}).get('attached_pic', 0) == 1:
                return 3
            return 0
        elif codec_type == constants.CODEC_AUDIO:
            return 1
        else:
            return 2

    result.sort(key=stream_sort_key)
    return result


def get_media_title_from_tags(input_info: dict) -> [None, str]:
    if input_info is None:
        return None
    return input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(constants.K_MEDIA_TITLE)


def get_media_title_from_filename(input_info: dict) -> [None, str]:
    if input_info is None:
        return None
    filename = input_info.get(constants.K_FORMAT, {}).get("filename")
    if not filename:
        return None
    base_filename = os.path.basename(filename)
    return ".".join(base_filename.split('.')[0:-1]).replace('_', ' ')


def should_replace_media_title(input_info: dict) -> bool:
    if constants.K_FORMAT not in input_info:
        # do not replace title if we didn't query for format
        return False
    filename_title = get_media_title_from_filename(input_info)
    if not filename_title:
        return False
    current_title = get_media_title_from_tags(input_info)
    if not current_title:
        return True
    year_matcher = r'\(\d{4}\)'
    if re.search(year_matcher, filename_title) and not re.search(year_matcher, current_title):
        return True
    if '_' in current_title:
        return True
    return False


def resolve_video_codec(desired_codec: [list, str], target_height: [None, int] = None,
                        video_info: [None, dict] = None) -> [None, str]:
    """
    Resolve to a real video codec as seen by the user (NOT the ffmpeg version). This may be adjusted based on the
    video info.
    :param desired_codec:
    :param target_height:
    :param video_info:
    :return:
    """
    result = __resolve_codec(desired_codec, video_info)

    # Choose based on video attributes
    if result is None:
        result = "h264"
        # my testing shows bitrate for h264 is lower than h265 and ~10x faster using software encoding
        if target_height and target_height >= 1080 and hwaccel.has_hw_codec('h265'):
            result = "h265"

    return result


def resolve_audio_codec(desired_codec: [list, str], audio_info: [None, dict] = None) -> [None, str]:
    """
    Resolve to a real audio codec as seen by the user (NOT the ffmpeg version). This may be adjusted based on the
    audio info.
    :param desired_codec:
    :param audio_info:
    :return:
    """
    result = __resolve_codec(desired_codec, audio_info)
    if result is None:
        result = "opus"
    return result


def __resolve_codec(desired_codec: [list, str], stream_info: [None, dict] = None) -> [None, str]:
    result = None

    if isinstance(desired_codec, list):
        current_codec = resolve_human_codec(None if stream_info is None else stream_info.get('codec_name', None))
        # check if current codec matches a desired codec
        for idx, codec in enumerate(desired_codec):
            codec = resolve_human_codec(codec)
            if codec == current_codec:
                result = codec
                break
        # prefer first codec
        if result is None and len(desired_codec) > 0:
            available = list(filter(lambda c: is_codec_available(c), desired_codec))
            if len(available) == 0:
                raise RuntimeError(f"No codecs available from desired list of {desired_codec}")
            result = available[0]
    else:
        result = desired_codec

    return resolve_human_codec(result)


def resolve_human_codec(codec: [None, str]) -> [None, str]:
    """
    Resolve human presented names of codecs
    :param codec: codec name or None
    :return:
    """
    result = None if codec is None else codec.lower().strip()
    if result == 'hevc':
        result = 'h265'
    elif result == 'avc':
        result = 'h264'
    elif result == 'vvc':
        result = 'h266'
    return result


def is_codec_available(codec: str) -> bool:
    """
    Check if a codec is available for use, in any sense. Some may required hardware encoding.
    """
    if hwaccel.require_hw_codec(codec):
        return hwaccel.has_hw_codec(codec)
    return True


def recommended_video_quality(target_height: int, target_video_codec: str) -> (int, int, int):
    """
    CRF
    https://slhck.info/video/2017/02/24/crf-guide.html
    Defaults to H265 the upcoming standard (2022), specifics for H264 the current most compatible
    H264:
    18 is visually lossless
    23 is the default for <= 720, 31 for 1080
    H265:
    28 for all resolutions

    :return: crf, bitrate, qp
    """
    # q=28 observed with libx264, crf=23
    qp = 28
    if target_height <= 480:
        if target_video_codec == 'h264':
            crf = 23
            bitrate = 600
        else:
            crf = 28
            bitrate = 440
    elif target_height <= 720:
        if target_video_codec == 'h264':
            crf = 23
            bitrate = 1200
        else:
            crf = 28
            bitrate = 700
    elif target_height <= 1080:
        # q=36 observed with libx264, crf=31
        qp = 34
        if target_video_codec == 'h264':
            crf = 31
            bitrate = 1100
        else:
            crf = 28
            # TODO: Avatar 1080p is 2055 with 2 5.1 streams
            bitrate = 1000
    else:
        # q=36 observed with libx264, crf=31
        qp = 34
        if target_video_codec == 'h264':
            crf = 31
            bitrate = 2500
        else:
            crf = 28
            bitrate = 1200

    return crf, bitrate, qp


def fps_video_filter(desired_frame_rate: [None, str, float]):
    if desired_frame_rate is None:
        return None
    return f"minterpolate=fps={desired_frame_rate}:mi_mode=blend"


def map_opus_audio_stream(arguments: list[str], audio_info: dict, audio_stream_idx: int, output_stream_spec: str,
                          audio_filters=None, force_stereo=False, mute_channels: [None, config.MuteChannels] = None):
    if audio_filters is None:
        audio_filters = []
    if mute_channels is None:
        mute_channels = config.get_global_config_mute_channels()

    audio_layout = None
    output_audio_layout = None

    if mute_channels == config.MuteChannels.VOICE and "volume=" in ",".join(audio_filters) and not force_stereo:
        audio_layout = tools.get_audio_layout_by_name(audio_info.get(constants.K_CHANNEL_LAYOUT, ''))
        # TODO: if we transcribe each channel individually, we can determine which channels to mute
        if audio_layout:
            if "(" in audio_layout.name:
                # opus does not support side or wide layouts
                output_audio_layout_name = re.sub(r'[(].*[)]', '', audio_layout.name)
            elif 2 < len(audio_layout.channels) < 5:
                # use ffmpeg upmix system to use more common 5.1 layout
                output_audio_layout_name = "5.1"
            else:
                output_audio_layout_name = None
            if output_audio_layout_name:
                output_audio_layout = tools.get_audio_layout_by_name(output_audio_layout_name)
            else:
                output_audio_layout = audio_layout
            logger.info("Audio layout found %s, output layout %s", audio_layout, output_audio_layout)
        else:
            logger.info("Muting all channels, audio layout not found %s",
                        audio_info.get(constants.K_CHANNEL_LAYOUT, ''))
            audio_layout = None
            output_audio_layout = None

    if audio_layout is not None and output_audio_layout is not None:
        mute_filter_complex = f"[{audio_stream_idx}:{audio_info[constants.K_STREAM_INDEX]}]channelsplit=channel_layout={audio_layout.name}"
        for c in audio_layout.channels:
            mute_filter_complex += f"[{c}]"
        mute_filter_complex += ';'
        for c in audio_layout.voice_channels:
            mute_filter_complex += f"[{c}]"
            mute_filter_complex += ",".join(audio_filters)
            mute_filter_complex += f"[{c}m];"
        for c in audio_layout.channels:
            if c in audio_layout.voice_channels:
                mute_filter_complex += f"[{c}m]"
            else:
                mute_filter_complex += f"[{c}]"
        mute_filter_complex += f"amerge=inputs={len(audio_layout.channels)},pan={output_audio_layout.name}"
        audio_mapping = audio_layout.map_to(output_audio_layout)
        for out_ch in output_audio_layout.channels:
            m = audio_mapping.get(out_ch)
            if m:
                mute_filter_complex += f"|{out_ch}"
                if len(m) > 1:
                    mute_filter_complex += "<"
                else:
                    mute_filter_complex += "="
                for i, in_ch in enumerate(m):
                    in_ch_idx = [i for i, value in enumerate(audio_layout.channels) if value == in_ch]
                    if i > 0:
                        mute_filter_complex += '+'
                    mute_filter_complex += f"c{in_ch_idx[0]}"
        mute_filter_complex += f'[afiltered]'
        arguments.extend(["-filter_complex", mute_filter_complex, "-map", "[afiltered]"])
    else:
        arguments.extend(["-map", f"{audio_stream_idx}:{audio_info[constants.K_STREAM_INDEX]}"])

        if force_stereo:
            arguments.extend([f"-ac:{output_stream_spec}", "2"])
        elif audio_info.get('channel_layout') == '5.1(side)':
            audio_filters.insert(0, "channelmap=channel_layout=5.1")
        elif audio_info.get('channel_layout') == '5.0(side)':
            audio_filters.insert(0, "channelmap=channel_layout=5.0")
        elif audio_info.get('channel_layout') == '7.1(wide)':
            audio_filters.insert(0, "channelmap=channel_layout=7.1")
        elif audio_info.get('channel_layout') == '4.0':
            # use ffmpeg up mix system to use more common 5.1 layout
            arguments.extend([f"-ac:{output_stream_spec}", "6"])

        if audio_filters:
            arguments.extend([f"-filter:{output_stream_spec}", ",".join(audio_filters)])

    arguments.extend([f"-c:{output_stream_spec}", hwaccel.ffmpeg_sw_codec("opus")])

    arguments.extend([f"-vbr:{output_stream_spec}", "on"])

    # enables optimizations in opus v1.1+
    if force_stereo or audio_info.get('channel_layout') in ['mono', 'stereo', '1.0', '2.0']:
        arguments.extend([f"-mapping_family:{output_stream_spec}", "0"])
    else:
        arguments.extend([f"-mapping_family:{output_stream_spec}", "1"])

    # use original bit rate if lower than default
    target_bitrate = None
    channels = audio_info[constants.K_CHANNELS]
    audio_bitrate = int(audio_info[constants.K_BIT_RATE]) if constants.K_BIT_RATE in audio_info else None
    if audio_bitrate is not None:
        if channels == 1 and audio_bitrate < (64 * 1024):
            target_bitrate = max(6, audio_bitrate)
        elif channels == 2 and audio_bitrate < (96 * 1024):
            target_bitrate = max(6, audio_bitrate)
        elif channels == 6 and audio_bitrate < (320 * 1024):
            target_bitrate = max(6, audio_bitrate)

    if target_bitrate is not None:
        arguments.extend([f"-b:{output_stream_spec}", str(target_bitrate)])


class EdlType(Enum):
    CUT = 0
    MUTE = 1
    SCENE = 2
    COMMERCIAL = 3
    BACKGROUND_BLUR = 4


class EdlEvent(object):

    def __init__(self, start, end, event_type: EdlType, title: str = None):
        self.start = start
        self.end = end
        self.event_type = event_type
        self.title = title

    def __repr__(self):
        if self.title:
            return f"{self.title}: {self.start} - {self.end}"
        else:
            return f"{self.start} - {self.end}"

    def length(self):
        return max(0, self.end - self.start)


def parse_edl(filename) -> list[EdlEvent]:
    events = []
    with open(filename, 'r') as edl_fd:
        for edl_line in edl_fd.readlines():
            if edl_line.startswith('##'):
                continue
            parts = edl_line.replace('-->', ' ').split(maxsplit=3)
            if len(parts) < 3:
                continue
            start = parse_edl_ts(parts[0])
            end = parse_edl_ts(parts[1])
            if parts[2].lower() in ('0', 'cut'):
                event_type = EdlType.CUT
            elif parts[2].lower() in ('1', 'mute'):
                event_type = EdlType.MUTE
            elif parts[2].lower() in ('2', 'scene'):
                event_type = EdlType.SCENE
            elif parts[2].lower() in ('3', 'com', 'commercial'):
                event_type = EdlType.COMMERCIAL
            elif parts[2].lower() in ('4', 'blur'):
                event_type = EdlType.BACKGROUND_BLUR
            else:
                raise Exception(f"Unknown EDL type: {parts[2]}")
            events.append(EdlEvent(start, end, event_type, parts[3].strip() if 3 < len(parts) else None))
    events.sort(key=lambda e: e.start)
    return events


def parse_edl_cuts(filename) -> list[EdlEvent]:
    return list(filter(lambda e: e.event_type in [EdlType.CUT, EdlType.COMMERCIAL], parse_edl(filename)))


def parse_edl_ts(s: str) -> float:
    if ":" in s:
        parts = list(s.split(":"))
        while len(parts) < 3:
            parts.insert(0, "0")
        return round(float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2].replace(',', '.')), 3)
    return round(float(s), 3)


def s_to_ts(t: float) -> str:
    hour = int(t / 3600.0)
    minute = int(t / 60) % 60
    second = t % 60.0
    return f"{hour:02d}:{minute:02d}:{second:06.3f}"


def write_mkv_tags(tags, filepath) -> None:
    """
    Writes the dictionary to an XML file for use my mkvpropedit.
    :param tags: name-value strings of tags
    :param filepath: path-like object for the file, will be overwritten
    """
    root = ET.Element("Tags")
    tag = ET.Element("Tag")
    root.append(tag)
    for k, v in tags.items():
        simple = ET.Element("Simple")
        tag.append(simple)
        name = ET.Element("Name")
        name.text = str(k)
        simple.append(name)
        value = ET.Element("String")
        value.text = str(v)
        simple.append(value)
    with open(filepath, "wb") as f:
        ET.ElementTree(root).write(f)


def filepath_is_mkv(filepath):
    filename = os.path.basename(filepath)
    return filename.endswith(".mkv") and not filename.startswith('.')


def filter_for_mkv(l):
    return filter(lambda f: filepath_is_mkv(f), l)


def split_every(n, iterable):
    return [iterable[i:i + n] for i in range(0, len(iterable), n)]


def check_already_running():
    """
    Check if other processes like me are running.
    :return: True if there are other processes, False if I'm the only one.
    """
    my_name = os.path.basename(sys.argv[0])
    others = []
    for p in psutil.process_iter():
        try:
            if my_name in " ".join(p.cmdline()) and p.pid != os.getpid():
                others.append(p)
        except (PermissionError, AccessDenied, ProcessLookupError, NoSuchProcess):
            pass
    if len(others) > 0:
        logger.error(f"process(es) already running: {list(map(lambda p: p.pid, others))}")
        return True
    return False


def core_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count())


def should_stop_processing():
    """
    Check if system resource usage is such that we should stop processing.
    :return: True if we should stop processing
    """

    # check load
    if os.getloadavg()[0] > 8:
        return True

    # check if we're on battery
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged:
        return True

    return False


def should_start_processing():
    """
    Check if system resource usage is such that we should start processing.
    :return: True if we should start processing
    """

    # check load
    if os.getloadavg()[0] >= 4:
        return False

    # check if we're on battery
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged:
        return False

    return True


def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d = {'_frame': frame}  # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)


def dumpstacks(signal, frame):
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    trace = []
    for threadId, stack in sys._current_frames().items():
        trace.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            trace.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                trace.append("  %s" % (line.strip()))
    print("\n".join(trace))


def setup_debugging():
    signal.signal(signal.SIGUSR1, debug)
    signal.signal(signal.SIGQUIT, dumpstacks)
    logger.debug("__debug__ is %s", __debug__)


def setup_logging():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)


def setup_cli():
    setup_logging()
    setup_debugging()


def remove_extension(path):
    if path is None:
        return None
    return os.path.splitext(path)[0]


def replace_extension(path, new_extension):
    if path is None:
        return None
    return os.path.splitext(path)[0] + '.' + new_extension


VIDEO_EXTENSIONS = ["mkv", "ts", "mp4", "mov"]


def is_video_file(filepath) -> bool:
    if os.path.basename(filepath).startswith('._'):
        # Apple metainfo
        return False
    if len(list(filter(lambda e: filepath.endswith('.' + e), VIDEO_EXTENSIONS))) > 0:
        return True

    try:
        input_info = json.loads(tools.ffprobe.check_output(
            ['-v', 'quiet', '-analyzeduration', ANALYZE_DURATION, '-probesize', PROBE_SIZE, '-print_format',
             'json', '-show_format', '-show_streams', filepath]))
        return len(list(filter(lambda s: s[constants.K_CODEC_TYPE] == constants.CODEC_VIDEO,
                               input_info.get('streams', [])))) > 0 and len(
            list(filter(lambda s: s[constants.K_CODEC_TYPE] == constants.CODEC_AUDIO,
                        input_info.get('streams', [])))) > 0
    except subprocess.CalledProcessError:
        return False


def generate_video_files(args, suffix=".mkv", fail_on_missing=False):
    """
    Searches for mkv files from the arguments that can be files and directories. A special case is two arguments, a
    mkv file and a missing second argument specifying the output file.
    :param args: list of files and/or directories
    :param suffix: the file suffix, default ".mkv"
    :param fail_on_missing: True to fail if an argument is specified that does not exist
    :return: yields a tuple of (input file, output file)
    """
    logger.debug("generate_video_files for suffix %s in paths %s", suffix, args)
    if len(args) == 2:
        # check special case of input file and output file
        if os.path.isfile(args[0]) and (os.path.isfile(args[1]) or not os.path.exists(args[1])):
            yield args[0], args[1]
            return

    for arg in args:
        if fail_on_missing and not os.path.exists(arg):
            raise FileNotFoundError(arg)
        if os.path.isfile(arg):
            yield arg, arg
        for root, dirs, files in os.walk(arg):
            for file in files:
                filepath = os.path.join(root, file)
                filename = os.path.basename(filepath)
                if filename.startswith('.'):
                    # skip hidden files
                    continue
                if suffix:
                    if not filename.endswith(suffix):
                        continue
                elif not is_video_file(filepath):
                    continue
                yield filepath, filepath


def filter_video_files(root, files, suffix=".mkv") -> list[str]:
    result = []
    for file in files:
        filepath = os.path.join(root, file)
        filename = os.path.basename(filepath)
        if filename.startswith('.'):
            # skip hidden files
            continue
        if suffix:
            if not filename.endswith(suffix):
                continue
        elif not is_video_file(filepath):
            continue
        result.append(filepath)
    return result


class ReturnCodeReducer:
    def __init__(self):
        self._code = 0

    def callback(self, c):
        if c != 0 and self._code == 0:
            self._code = c

    def code(self) -> int:
        return self._code


def error_callback_dump(e):
    if type(e) == KeyboardInterrupt:
        logger.error("User interrupt")
    elif type(e) == StopIteration:
        logger.error("Stopped: %s", str(e))
    else:
        logger.error(str(e), exc_info=e)


def seconds_to_timespec(seconds):
    prefix = ""
    if seconds < 0:
        prefix = "-"
        seconds = abs(seconds)
    return prefix + str(datetime.timedelta(seconds=seconds))


def edl_for_video(filepath):
    edl_path = replace_extension(filepath, 'edl')
    if os.path.exists(edl_path):
        return edl_path
    edl_bak_path = replace_extension(filepath, 'bak.edl')
    if os.path.exists(edl_bak_path):
        return edl_bak_path
    return edl_path


def __is_commercial_chapter(chapter_info: dict):
    end_time = float(chapter_info['end_time'])
    start_time = float(chapter_info['start_time'])
    if start_time < 0:
        return False
    if end_time - start_time < 1:
        return False
    return "Commercial" in chapter_info.get("tags", {}).get("title", "")


def has_chapters_from_source_media(input_info) -> (bool, list[dict]):
    """
    chapters without "Commercial " are only there from ripped media or after cutting commercials
    :param input_info:
    :return: True if there are chapters and none are commercials, list of commercial chapter info
    """
    chapters = input_info.get(constants.K_CHAPTERS, [])
    chapters_commercials = list(filter(__is_commercial_chapter, chapters))
    return len(chapters) > 0 and len(chapters_commercials) == 0, chapters_commercials


def round_duration(duration):
    # broadcasts are usually increments of 30 minutes
    half_hours = int(duration / 1800)
    if duration % 1800 > 600:
        half_hours += 1
    return half_hours * 1800


def round_episode_duration(video_info):
    return round_duration(episode_info(video_info)[1])


def is_from_dvr(input_info):
    """
    Determine if the file is likely to have been recorded from OTA broadcast.
    :param input_info:
    :return:
    """
    if is_truthy(input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(constants.K_COMSKIP_SKIP)):
        return False
    if has_chapters_from_source_media(input_info)[0]:
        return False

    duration = float(input_info[constants.K_FORMAT][constants.K_DURATION])
    rounded_duration = round_duration(duration)
    already_cut = (100 * abs(rounded_duration - duration) / duration) > 15
    return not already_cut


def get_arguments_from_config(argv, filename) -> list[str]:
    """
    Read command line arguments from files stored in the filename searching:
    1. path tree starting at any file or directory argument in 'argv'
    2. home directory with any leading '.' removed from the filename
    3. /etc directory with any leading '.' removed from the filename
    :param argv:
    :param filename:
    :return:
    """
    path = None
    for a in argv:
        p = os.path.realpath(a)
        if os.path.exists(p):
            path = p
            break
    files = []
    if path:
        if os.path.isdir(path):
            files.append(os.path.join(path, filename))
        while True:
            p = os.path.dirname(path)
            if not p or p == path:
                break
            files.append(os.path.join(p, filename))
            path = p
    etc_filename = re.sub('^[.]', '', filename)
    files.append(f"{os.environ['HOME']}/{etc_filename}")
    files.append(f"/etc/{etc_filename}")
    logger.debug(f"Looking for arguments in {','.join(files)}")

    for conf in files:
        if os.path.isfile(conf):
            arguments = []
            with open(conf, "r") as text_file:
                for line in text_file.readlines():
                    line = line.strip()
                    if len(line) > 0 and not line.startswith('#'):
                        arguments.append(line)
                logger.info(f"Read arguments from {conf}: {' '.join(arguments)}")
            return arguments

    return []


def episode_info(input_info: dict) -> (int, float, float):
    """
    :param input_info:
    :return: number of episodes, episode duration (seconds), total duration (seconds)
    """
    duration = float(input_info[constants.K_FORMAT]['duration'])
    episode_duration = duration
    episode_count = 1
    filename = input_info[constants.K_FORMAT]['filename']
    if filename:
        episode_range = re.search(r"E(\d+)-E(\d+)", filename)
        if episode_range:
            count = (int(episode_range.group(2)) - int(episode_range.group(1))) + 1
            if count > 0:
                episode_count = count
                episode_duration /= count

    return episode_count, episode_duration, duration


def get_common_episode_duration(video_infos: list[dict]):
    """
    Get the most common duration from the list of videos.
    :param video_infos: list of info dicts from ffprobe
    :return:
    """
    if len(video_infos) == 0:
        return None
    stats = dict()
    for video_info in video_infos:
        episode_count, episode_duration, duration = episode_info(video_info)
        episode_duration = round_duration(episode_duration)
        if episode_duration in stats:
            stats[episode_duration] += 1
        else:
            stats[episode_duration] = 1
    stats_list = list(stats.items())
    stats_list.sort(key=lambda e: e[1], reverse=True)
    return stats_list[0][0]


def match_owner_and_perm(target_path: str, source_path: str) -> bool:
    result = True
    source_stat = os.stat(source_path)
    try:
        os.chown(target_path, source_stat.st_uid, source_stat.st_gid)
    except OSError:
        logger.warning(f"Changing ownership of {target_path} failed, continuing")
        result = False

    try:
        st_mode = source_stat.st_mode
        # if source is dir and has suid or guid and target is a file, mask suid/guid
        if os.path.isfile(target_path):
            st_mode &= ~(stat.S_ISUID | stat.S_ISGID)
        os.chmod(target_path, st_mode)
    except OSError:
        logger.warning(f"Changing permission of {target_path} failed, continuing")
        result = False

    return result


def is_truthy(value) -> bool:
    if value is None:
        return False
    return str(value).lower() in ['true', 't', 'yes', 'y', '1']


def frame_rate_from_s(frame_rate_s: Union[str, None]) -> Union[float, None]:
    if frame_rate_s is None:
        return None
    framerate: Union[float, None] = None
    frame_rate_s = frame_rate_s.lower()
    if frame_rate_s[0].isdigit():
        # remove suffix, like 'p'
        framerate = float(re.sub(r'\D', '', frame_rate_s))
    elif frame_rate_s in constants.FRAME_RATE_NAMES.keys():
        framerate = float(eval(constants.FRAME_RATE_NAMES[frame_rate_s]))
    else:
        logger.warning("Unknown framerate %s", frame_rate_s)
    return framerate


def should_adjust_frame_rate(current_frame_rate: Union[None, str, float], desired_frame_rate: Union[None, str, float],
                             tolerance: float = 0.25) -> bool:
    logger.debug("should_adjust_frame_rate(current_frame_rate=%s, desired_frame_rate=%s, tolerance=%s",
                 current_frame_rate, desired_frame_rate, str(tolerance))

    if current_frame_rate in [None, '', '0', '0/0'] or desired_frame_rate in [None, '', '0', '0/0']:
        return False

    if type(current_frame_rate) == str:
        current_frame_rate_f = eval(constants.FRAME_RATE_NAMES.get(current_frame_rate.lower(), current_frame_rate))
    else:
        current_frame_rate_f = current_frame_rate

    if type(desired_frame_rate) == str:
        desired_frame_rate_f = eval(constants.FRAME_RATE_NAMES.get(desired_frame_rate.lower(), desired_frame_rate))
    else:
        desired_frame_rate_f = desired_frame_rate

    frame_rate_pct = abs(current_frame_rate_f - desired_frame_rate_f) / max(current_frame_rate_f, desired_frame_rate_f)
    return frame_rate_pct >= tolerance
