import _thread
import atexit
import code
import datetime
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from configparser import NoOptionError
from enum import Enum
from multiprocessing import Pool
from typing import Union
from xml.etree import ElementTree as ET

import psutil
from psutil import AccessDenied, NoSuchProcess

from . import hwaccel, tools, config, constants, progress, procprofile
from .proc_invoker import StreamCapturingLogger, pre_flight_check
from .terminalui import terminalui_wrapper

_allocate_lock = _thread.allocate_lock
_once_lock = _allocate_lock()

logger = logging.getLogger(__name__)

TEMPFILENAMES = []

ANALYZE_DURATION = '20000000'
PROBE_SIZE = '20000000'

MEDIA_ROOTS: Union[list[str], None] = None


def get_media_roots() -> list[str]:
    global MEDIA_ROOTS
    if MEDIA_ROOTS is None:
        _once_lock.acquire()
        try:
            if MEDIA_ROOTS is None:
                MEDIA_ROOTS = _find_media_roots()
        finally:
            _once_lock.release()
    return MEDIA_ROOTS


def _find_media_roots() -> list[str]:
    try:
        paths = config.get_global_config_option('media', 'paths').split(',')
        if len(paths) > 0:
            mount_points = []
            for part in psutil.disk_partitions(all=True):
                if any([os.path.isdir(os.path.join(part.mountpoint, path)) for path in paths]):
                    mount_points.append(part.mountpoint)
            if mount_points:
                return mount_points
    except (KeyError, NoOptionError):
        logger.debug('No media.paths in config')
    try:
        roots = []
        for root in config.get_global_config_option('media', 'root').split(','):
            root = root.replace('$HOME', os.environ['HOME'])
            if os.path.isdir(root):
                roots.append(root)
        if roots:
            return roots
    except (KeyError, NoOptionError):
        logger.debug('No media.root in config')
    logger.info('No valid media.root found, returning user home')
    return [os.environ['HOME']]


def get_media_paths(roots=None, paths=None):
    if roots is None:
        roots = get_media_roots()
    elif isinstance(roots, (list, set)):
        roots = list(roots)
    else:
        roots = [roots]

    if paths is None:
        paths = config.get_global_config_option('media', 'paths').split(',')
    elif isinstance(paths, (list, set)):
        paths = list(paths)
    else:
        paths = [paths]

    result = set()
    for root in roots:
        for path in paths:
            if os.path.isabs(path):
                joined = path
            else:
                joined = os.path.join(root, path)
            if os.path.exists(joined):
                result.add(joined)
    return list(result)


def get_media_file_relative_to_root(file_name: str, roots: list[str]) -> tuple[str, str]:
    if not roots:
        return file_name, '/'
    for root in roots:
        if root.endswith('/'):
            prefix = root
        else:
            prefix = root + '/'
        if file_name.startswith(prefix):
            return file_name[len(prefix):], prefix
    return file_name, '/'


def fatal(message):
    logger.critical(message)
    sys.exit(255)


def exception_hook(exctype, value, traceback_obj):
    if isinstance(value, KeyboardInterrupt):
        logger.critical("user interrupt")
        sys.exit(130)
    if isinstance(value, subprocess.CalledProcessError):
        logger.critical("subprocess error", exc_info=(exctype, value, traceback_obj))
        if value.stdout:
            logger.critical("stdout: %s", value.stdout)
        if value.stderr:
            logger.critical("stderr: %s", value.stderr)
    else:
        logger.critical("Uncaught exception", exc_info=(exctype, value, traceback_obj))
    sys.exit(255)


sys.excepthook = exception_hook


def finish():
    global TEMPFILENAMES
    while TEMPFILENAMES:
        fn = TEMPFILENAMES.pop()
        try:
            if os.path.isfile(fn):
                os.remove(fn)
        except FileNotFoundError:
            pass


def get_plex_url():
    return config.get_global_config_option('plex', 'url', fallback=None)


_KEYFRAMES_PATTERN = re.compile(r'[\d.]+')


def load_keyframes_by_seconds(filepath) -> list[float]:
    # "-skip_frame nointra" brings in too many frames in mpeg2video
    ffprobe_keyframes = ["-loglevel", "error", "-skip_frame", "nokey", "-select_streams", "v:0",
                         "-show_entries",
                         "frame=pkt_pts_time" if int(tools.ffprobe.version) == 4 else "frame=pts_time",
                         "-of", "csv=print_section=0", filepath]
    keyframes = list(map(lambda s: float(_KEYFRAMES_PATTERN.search(s)[0]),
                         filter(lambda s: len(s) > 0,
                                tools.ffprobe.check_output(ffprobe_keyframes, universal_newlines=True,
                                                           text=True, stderr=subprocess.DEVNULL).splitlines())))
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


def find_input_info(filename, raise_errors=True) -> dict:
    try:
        input_info = json.loads(tools.ffprobe.check_output(
            ['-v', 'quiet', '-analyzeduration', ANALYZE_DURATION, '-probesize', PROBE_SIZE, '-print_format',
             'json',
             '-show_format', '-show_streams', '-show_chapters', filename]))
    except subprocess.CalledProcessError as cpe:
        if raise_errors:
            raise cpe
        else:
            return dict()
    fix_closed_caption_report(input_info, filename)
    return input_info


def get_video_height(video_info: dict) -> [None, int]:
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
    elif height == '4k':
        height = 2160
    elif height == '8k':
        height = 4320
    return height


def get_video_width(video_info: dict) -> [None, int]:
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


def get_video_width_height(video_info: dict) -> [None, tuple[int, int]]:
    """
    Get the (width, height) of the video considering symbolic names.
    :param video_info: can be all of the input_info to use default video stream, or a single video stream
    :return: int of width or None
    """
    if 'streams' in video_info:
        video_info = find_video_stream(video_info)
    width = video_info['width']
    if not width:
        return None
    height = video_info['height']
    if not height:
        return None
    return width, height


def get_video_depth(video_info: dict) -> [None, int]:
    """
    Get the bit depth of the video.
    :param video_info: can be all of the input_info to use default video stream, or a single video stream
    :return: int: 8, 10, etc. or None
    """
    if 'streams' in video_info:
        video_info = find_video_stream(video_info)
    profile = video_info.get('profile', None)
    if not profile:
        return None
    if '10' in profile:
        return 10
    return 8


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
    global TEMPFILENAMES
    if tempfilename is None:
        dirname = os.path.dirname(os.path.abspath(input_file))
        base = os.path.basename(input_file)
        parts = base.split('.')
        tempfilename = os.path.join(dirname, '.~' + '.'.join(parts[0:-1]) + '.transcoded.' + parts[-1])
    if os.path.isfile(tempfilename):
        if (time.time() - os.path.getmtime(tempfilename)) < 172800:
            logger.info(f"Already transcoding, skipping {input_file} ({tempfilename})")
            # We don't want clean up to remove these files and mess up other processes
            try:
                TEMPFILENAMES.remove(tempfilename)
            except ValueError:
                pass
            if exit:
                sys.exit(0)
            else:
                return 255
        else:
            os.remove(tempfilename)
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


year_matcher = re.compile(r'\(\d{4}\)')


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
    if year_matcher.search(filename_title) and not year_matcher.search(current_title):
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


def recommended_video_quality(target_height: int, target_video_codec: str, bit_depth: Union[int, None]) -> (
int, int, int):
    """
    # FIXME: bit-rates need to be revisited
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
        qp = 30
        if target_video_codec == 'h264':
            crf = 31
            bitrate = 1100
        else:
            crf = 28
            # TODO: Avatar 1080p is 2055 with 2 5.1 streams
            bitrate = 1200
    else:
        # q=36 observed with libx264, crf=31
        qp = 30
        if target_video_codec == 'h264':
            crf = 31
            bitrate = 2500
        else:
            crf = 28
            bitrate = 1700

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
        mute_filter_complex += '[afiltered]'
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


def s_to_ts(t: float) -> str:
    hour = int(t / 3600.0)
    minute = int(t / 60) % 60
    second = t % 60.0
    return f"{hour:02d}:{minute:02d}:{second:06.3f}"


def ms_to_ts(t: float) -> str:
    return s_to_ts(t / 1000.0)


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
        ET.ElementTree(root).write(f, 'ISO-8859-1')


def filepath_is_mkv(filepath):
    filename = os.path.basename(filepath)
    return filename.endswith(".mkv") and not filename.startswith('.')


def filter_for_mkv(file_list):
    return filter(lambda f: filepath_is_mkv(f), file_list)


def split_every(n, iterable):
    return [iterable[i:i + n] for i in range(0, len(iterable), n)]


def check_already_running(quiet=False):
    """
    Check if other processes like me are running.
    :return: True if there are other processes, False if I'm the only one.
    """
    my_name = os.path.basename(sys.argv[0])
    others = []
    for p in psutil.process_iter():
        try:
            if p.pid != os.getpid():
                cmdline = list(p.cmdline())
                if len(cmdline) > 0:
                    if 'python' in os.path.basename(cmdline[0]).lower():
                        cmdline.pop(0)
                if len(cmdline) > 0 and my_name in cmdline[0]:
                    others.append(p)
        except (PermissionError, AccessDenied, ProcessLookupError, NoSuchProcess):
            pass
    if len(others) > 0:
        if not quiet:
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


def setup_logging(level=logging.INFO):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s', level=level, force=True)


def setup_cli(level=logging.INFO, start_gauges=True):
    pre_flight_check()
    setup_logging(level)
    setup_debugging()
    if start_gauges:
        progress.start_compute_gauges()


_CLI_WRAPPED: bool = False


def cli_wrapper(func, *args, **kwargs):
    global _CLI_WRAPPED
    try:
        if _CLI_WRAPPED:
            return func(*args, **kwargs)
        _CLI_WRAPPED = True

        if not args and not kwargs:
            cli = True
            args = sys.argv[1:]
        else:
            cli = False

        no_curses_opt = False
        if '--no-curses' in args:
            no_curses_opt = True
            args = list(filter(lambda e: e != '--no-curses', args))
        if kwargs:
            no_curses_opt = kwargs.get('no_curses', False) or kwargs.get('no-curses', False)
            kwargs = kwargs.copy()
            kwargs.pop('no_curses', '')
            kwargs.pop('no-curses', '')

        def wrapped_func() -> int:
            try:
                atexit.register(finish)
                procprofile.memory_monitor_start()
                if cli:
                    return func(args)
                else:
                    return func(*args, **kwargs)
            finally:
                procprofile.memory_monitor_stop()

        use_curses = sys.stdout.isatty() and not no_curses_opt
        if use_curses:
            sys.exit(terminalui_wrapper(wrapped_func))
        else:
            setup_cli()
            sys.exit(wrapped_func())

    finally:
        _CLI_WRAPPED = False


class PoolApplyWrapper:

    def __init__(self, func):
        self.func = func
        self.rootLogLevel = logging.getLogger().level
        self.progress_queue = progress.setup_parent_progress()

    def __call__(self, *args, **kwargs):
        progress.setup_subprocess_progress(self.progress_queue, self.rootLogLevel)
        stdout = StreamCapturingLogger('stdout', logger, logging.INFO)
        stderr = StreamCapturingLogger('stderr', logger, logging.ERROR)
        procprofile.memory_monitor_start()
        try:
            return self.func(*args, **kwargs)
        finally:
            stdout.finish()
            stderr.finish()
            procprofile.memory_monitor_stop()


def pool_apply_wrapper(func):
    """
    This wrapper forwards subprocess logging and progress to the parent process.
    :param func: the function to wrap
    :return: func result
    """
    return PoolApplyWrapper(func)


def pool_join_with_timeout(pool: Pool, timeout: float = 5 * 60):
    pool.join()


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

    def set_code(self, c: int):
        self._code = c


def error_callback_dump(e):
    if isinstance(e, KeyboardInterrupt):
        logger.error("User interrupt")
    elif isinstance(e, StopIteration):
        logger.error("Stopped: %s", str(e))
    elif isinstance(e, subprocess.CalledProcessError):
        logger.error("%s\n%s\n%s", str(e), e.stdout, e.stderr, exc_info=e)
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


def is_ripped_from_media(input_info: dict) -> bool:
    if not input_info:
        return False
    tag_values = list()
    tag_values.extend(input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).values())
    for stream in input_info[constants.K_STREAMS]:
        tag_values.extend(stream.get(constants.K_TAGS, {}).values())
    if any(filter(lambda v: 'makemkv' in v.lower(), tag_values)):
        return True
    return False


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


EPISODE_PATTERN = re.compile(r"E(\d+)-E(\d+)")


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
        episode_range = EPISODE_PATTERN.search(filename)
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


def is_truthy(value) -> bool:
    if value is None:
        return False
    return str(value).lower() in ['true', 't', 'yes', 'y', '1']


_FRAME_RATE_INVALID_CHARS = re.compile(r'\D')


def frame_rate_from_s(frame_rate_s: Union[str, None]) -> Union[float, None]:
    if frame_rate_s is None:
        return None
    frame_rate: Union[float, None] = None
    frame_rate_s = frame_rate_s.lower()
    if frame_rate_s[0].isdigit():
        # remove suffix, like 'p'
        frame_rate = float(_FRAME_RATE_INVALID_CHARS.sub('', frame_rate_s))
    elif frame_rate_s in constants.FRAME_RATE_NAMES.keys():
        frame_rate = float(eval(constants.FRAME_RATE_NAMES[frame_rate_s]))
    else:
        logger.warning("Unknown framerate %s", frame_rate_s)
    return frame_rate


def should_adjust_frame_rate(current_frame_rate: Union[None, str, float], desired_frame_rate: Union[None, str, float],
                             tolerance: float = 0.25) -> bool:
    logger.debug("should_adjust_frame_rate(current_frame_rate=%s, desired_frame_rate=%s, tolerance=%s",
                 current_frame_rate, desired_frame_rate, str(tolerance))

    if current_frame_rate in [None, '', '0', '0/0'] or desired_frame_rate in [None, '', '0', '0/0']:
        return False

    if isinstance(current_frame_rate, str):
        current_frame_rate_f = eval(constants.FRAME_RATE_NAMES.get(current_frame_rate.lower(), current_frame_rate))
    else:
        current_frame_rate_f = current_frame_rate

    if isinstance(desired_frame_rate, str):
        desired_frame_rate_f = eval(constants.FRAME_RATE_NAMES.get(desired_frame_rate.lower(), desired_frame_rate))
    else:
        desired_frame_rate_f = desired_frame_rate

    frame_rate_pct = abs(current_frame_rate_f - desired_frame_rate_f) / max(current_frame_rate_f, desired_frame_rate_f)
    return frame_rate_pct >= tolerance


def is_file_in_hidden_dir(filepath) -> bool:
    """
    Check if the file is inside a hidden directory, i.e. a directory name starting
    with ".". Any directory in the path starting with "." will return True.
    :param filepath: the full or partial path
    :return:
    """
    return filepath.startswith('.') or f"{os.path.sep}." in filepath
