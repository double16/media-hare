import _thread
import code
import configparser
import datetime
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from enum import Enum
from shutil import which

import psutil
from psutil import AccessDenied

from . import tools
from . import hwaccel

KILOBYTES_MULT = 1024
MEGABYTES_MULT = 1024 * 1024
GIGABYTES_MULT = 1024 * 1024 * 1024

_allocate_lock = _thread.allocate_lock
_once_lock = _allocate_lock()
_config_lock = _allocate_lock()

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

K_STREAM_TITLE = 'title'
K_STREAM_LANGUAGE = 'language'
K_STREAM_INDEX = 'index'
K_CODEC_NAME = 'codec_name'
K_CODEC_TYPE = 'codec_type'
K_FORMAT = 'format'
K_DURATION = 'duration'
K_TAGS = 'tags'
K_CHAPTERS = 'chapters'
K_BIT_RATE = 'bit_rate'
K_CHANNELS = 'channels'
K_DISPOSITION = 'disposition'
K_STREAMS = 'streams'
K_FILTER_VERSION = 'PFILTER_VERSION'
K_FILTER_HASH = 'PFILTER_HASH'
K_FILTER_SKIP = 'PFILTER_SKIP'
K_FILTER_STOPPED = 'PFILTER_STOPPED'
K_COMSKIP_HASH = 'COMSKIP_HASH'
K_AUDIO_TO_TEXT_VERSION = 'AUDIO2TEXT_VERSION'
K_MEDIA_TITLE = 'title'

CODEC_SUBTITLE_ASS = 'ass'
CODEC_SUBTITLE_SRT = 'srt'
CODEC_SUBTITLE_SUBRIP = 'subrip'
CODEC_SUBTITLE_TEXT_BASED = [CODEC_SUBTITLE_ASS, CODEC_SUBTITLE_SRT, CODEC_SUBTITLE_SUBRIP]
CODEC_SUBTITLE_DVDSUB = 'dvd_subtitle'
CODEC_SUBTITLE_BLURAY = 'hdmv_pgs_subtitle'
LANGUAGE_ENGLISH = 'eng'
CODEC_SUBTITLE = 'subtitle'
CODEC_AUDIO = 'audio'
CODEC_VIDEO = 'video'

FRAME_RATE_NAMES = {'ntsc': '30000/1001', 'pal': '25.0', 'film': '24.0', 'ntsc_film': '24000/1001'}

TITLE_ORIGINAL = 'Original'
TITLE_FILTERED = 'Filtered'
TITLE_FILTERED_FORCED = 'Filtered Only'

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
    paths = get_global_config_option('media', 'paths').split(',')
    if len(paths) > 0:
        for p in psutil.disk_partitions(all=True):
            if os.path.isdir(os.path.join(p.mountpoint, paths[0])):
                return p.mountpoint
    for root in get_global_config_option('media', 'root').split(','):
        root = root.replace('$HOME', os.environ['HOME'])
        if os.path.isdir(root):
            return root
    raise FileNotFoundError('No media.root in config')


def get_media_paths(base=None):
    if base is None:
        base = get_media_base()
    paths = get_global_config_option('media', 'paths').split(',')
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


def find_ffmpeg():
    return tools.find_ffmpeg()


def find_ffprobe():
    return tools.find_ffprobe()

comskip_path = None


def find_comskip():
    global comskip_path
    if comskip_path is None:
        _once_lock.acquire()
        try:
            if comskip_path is None:
                comskip_path = _find_comskip()
        finally:
            _once_lock.release()
    return comskip_path


def _find_comskip():
    comskip = which("comskip")
    if not os.access(comskip, os.X_OK):
        fatal(f"{comskip} is not an executable")
    return comskip


comskip_gui_path = None


def find_comskip_gui():
    global comskip_gui_path
    if comskip_gui_path is None:
        _once_lock.acquire()
        try:
            if comskip_gui_path is None:
                comskip_gui_path = _find_comskip_gui()
        finally:
            _once_lock.release()
    return comskip_gui_path


def _find_comskip_gui():
    comskip_gui = which("comskip-gui")
    if not os.access(comskip_gui, os.X_OK):
        fatal(f"{comskip_gui} is not an executable")
    return comskip_gui


mkvpropedit_path = None


def find_mkvpropedit():
    global mkvpropedit_path
    if mkvpropedit_path is None:
        _once_lock.acquire()
        try:
            if mkvpropedit_path is None:
                mkvpropedit_path = _find_mkvpropedit()
        finally:
            _once_lock.release()
    return mkvpropedit_path


def _find_mkvpropedit():
    mkvpropedit = which("mkvpropedit")
    if not os.access(mkvpropedit, os.X_OK):
        fatal(f"{mkvpropedit} is not an executable")
    return mkvpropedit


vosk_path = None


def find_vosk():
    global vosk_path
    if vosk_path is None:
        _once_lock.acquire()
        try:
            if vosk_path is None:
                vosk_path = _find_vosk()
        finally:
            _once_lock.release()
    return vosk_path


def _find_vosk():
    vosk = which("vosk-transcriber")
    if not os.access(vosk, os.X_OK):
        fatal(f"{vosk} is not an executable")
    return vosk


def get_plex_url():
    return get_global_config_option('plex', 'url', fallback=None)


def load_keyframes_by_seconds(filepath) -> list[float]:
    ffprobe_keyframes = [find_ffprobe(), "-loglevel", "error", "-skip_frame", "nointra", "-select_streams", "v:0",
                         "-show_entries",
                         "frame=pkt_pts_time" if int(tools.ffprobe_version) == 4 else "frame=pts_time",
                         "-of", "csv=print_section=0", filepath]
    keyframes = list(map(lambda s: float(re.search(r'[\d.]+', s)[0]),
                         filter(lambda s: len(s) > 0,
                                subprocess.check_output(ffprobe_keyframes, universal_newlines=True,
                                                        text=True).splitlines())))
    if len(keyframes) == 0:
        raise ChildProcessError("No key frames returned, suspect ffprobe command line is broken")

    return keyframes


class KeyframeSearchPreference(Enum):
    CLOSEST = 0
    AFTER = 1
    BEFORE = 2


def find_desired_keyframe(keyframes: list[float], target_time: float,
                          search_preference: KeyframeSearchPreference = KeyframeSearchPreference.CLOSEST) -> float:
    """
    Find the desired keyframe for the target_time.
    :param keyframes: list of keyframes in the video with times directly from the video
    :param target_time: time we want, zero based, because it's a human-readable time
    :param search_preference:
    :return:
    """
    if len(keyframes) == 0:
        return target_time

    return _find_desired_keyframe(keyframes, target_time, keyframes[0], search_preference)


def _find_desired_keyframe(keyframes: list[float], target_time: float, start_time: float,
                           search_preference: KeyframeSearchPreference = KeyframeSearchPreference.CLOSEST,
                           ) -> float:
    if len(keyframes) == 0:
        return target_time
    elif len(keyframes) == 1:
        return keyframes[0]
    elif len(keyframes) == 2:
        if search_preference == KeyframeSearchPreference.AFTER:
            if _keyframe_compare(keyframes[0], target_time, start_time) > 0:
                return keyframes[0]
            return keyframes[1]
        elif search_preference == KeyframeSearchPreference.BEFORE:
            if _keyframe_compare(keyframes[0], target_time, start_time) < 0:
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


def fix_closed_caption_report(input_info, ffprobe, filename):
    video_info = find_video_stream(input_info)
    if video_info and 'Closed Captions' in subprocess.check_output(
            [ffprobe, '-analyzeduration', ANALYZE_DURATION, '-probesize', PROBE_SIZE, filename],
            stderr=subprocess.STDOUT, text=True):
        video_info['closed_captions'] = 1


def find_input_info(filename):
    ffprobe = find_ffprobe()
    input_info = json.loads(subprocess.check_output(
        [ffprobe, '-v', 'quiet', '-analyzeduration', ANALYZE_DURATION, '-probesize', PROBE_SIZE, '-print_format',
         'json',
         '-show_format', '-show_streams', '-show_chapters', filename]))
    fix_closed_caption_report(input_info, ffprobe, filename)
    return input_info


def get_video_height(video_info):
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


def find_original_and_filtered_streams(input_info, codec_type, codec_names=None, language=None):
    original = None
    filtered = None
    filtered_forced = None

    streams = find_streams_by_codec_and_language(input_info, codec_type, codec_names, language)

    filtered_streams = list(
        filter(lambda stream: stream['tags'].get(K_STREAM_TITLE) == TITLE_FILTERED, streams))
    if (len(filtered_streams)) > 0:
        filtered = filtered_streams[0]

    filtered_forced_streams = list(
        filter(lambda stream: stream['tags'].get(K_STREAM_TITLE) == TITLE_FILTERED_FORCED, streams))
    if (len(filtered_forced_streams)) > 0:
        filtered_forced = filtered_forced_streams[0]

    original_streams = list(
        filter(lambda stream: stream['tags'].get(K_STREAM_TITLE) == TITLE_ORIGINAL, streams))
    if (len(original_streams)) > 0:
        original = original_streams[0]
    else:
        original_streams = list(
            filter(lambda stream: stream['tags'].get(K_STREAM_TITLE) != TITLE_FILTERED, streams))
        if (len(original_streams)) > 0:
            original = original_streams[0]

    return original, filtered, filtered_forced


def find_streams_by_codec_and_language(input_info, codec_type, codec_names=None, language=None):
    streams = list(filter(lambda stream: stream['codec_type'] == codec_type
                                         and (codec_names is None or stream['codec_name'] in codec_names)
                                         and stream['tags'].get('language') == language,
                          input_info['streams']))
    if len(streams) == 0:
        streams = list(filter(lambda stream: stream['codec_type'] == codec_type
                                             and (codec_names is None or stream['codec_name'] in codec_names)
                                             and stream['tags'].get('language') is None,
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
    return stream_info[K_CODEC_TYPE] == CODEC_VIDEO and (
            not stream_info.get(K_DISPOSITION) or stream_info.get(K_DISPOSITION).get('attached_pic') != 1)


def is_audio_stream(stream_info: dict) -> bool:
    return stream_info[K_CODEC_TYPE] == CODEC_AUDIO


def is_subtitle_text_stream(stream_info: dict) -> bool:
    if stream_info is None:
        return False
    return stream_info.get(K_CODEC_TYPE, '') == CODEC_SUBTITLE and stream_info.get(K_CODEC_NAME,
                                                                                   '') in CODEC_SUBTITLE_TEXT_BASED


def find_video_streams(input_info) -> list[dict]:
    """
    Find all video streams that do not have other purposes, such as attached pictures.
    :param input_info:
    :return: list of stream info maps
    """
    streams = list(filter(lambda stream: is_video_stream(stream), input_info[K_STREAMS]))
    return streams


def find_video_stream(input_info) -> dict:
    """
    Find the primary video stream.
    :param input_info:
    :return: stream map or None
    """
    streams = find_video_streams(input_info)
    english = find_english_streams(streams)
    if len(english) > 0:
        streams = english
    return streams[0]


def find_audio_streams(input_info):
    streams = list(filter(lambda stream: is_audio_stream(stream), input_info[K_STREAMS]))

    # Check for profanity filter streams
    filter_streams = list(
        filter(lambda s: s.get('tags', {}).get(K_STREAM_TITLE) in [TITLE_ORIGINAL, TITLE_FILTERED], streams))
    if len(filter_streams) == 2:
        return filter_streams

    # prefer English streams
    english = find_english_streams(streams)
    if len(english) > 0:
        streams = english
    if len(streams) < 2:
        return streams
    # Pick the largest bitrate
    if len(list(filter(lambda stream: K_BIT_RATE in stream, streams))) == len(streams):
        # In case all of the bit rates are the same, we don't want to lose the order when we leave this block
        streams2 = streams.copy()
        streams2.sort(key=lambda e: e[K_BIT_RATE], reverse=True)
        if streams2[0][K_BIT_RATE] > streams2[1][K_BIT_RATE]:
            return streams2[0:1]
    # Pick the most channels
    if len(list(filter(lambda stream: K_CHANNELS in stream, streams))) == len(streams):
        # In case all of the channel counts are the same, we don't want to lose the order when we leave this block
        streams2 = streams.copy()
        streams2.sort(key=lambda e: e[K_CHANNELS], reverse=True)
        if streams2[0][K_CHANNELS] > streams2[1][K_CHANNELS]:
            return streams2[0:1]
    # Pick default disposition
    default_streams = list(
        filter(lambda stream: stream.get(K_DISPOSITION) and stream.get(K_DISPOSITION).get('default') > 0, streams))
    if len(default_streams) > 0:
        return default_streams[0:1]

    return streams


def find_attached_pic_stream(input_info):
    return list(filter(lambda stream: (stream.get(K_DISPOSITION) and stream.get(K_DISPOSITION).get('attached_pic') > 0)
                                      or (stream.get(K_TAGS) and stream.get(K_TAGS).get('MIMETYPE', '').startswith('image/')),
                       input_info[K_STREAMS]))


def sort_streams(streams: list[dict]) -> list[dict]:
    """
    Sort streams in order of video, audio, other. In case of using complex filter graphs, the outputs will be ordered
    first regardless of stream mapping.
    :param streams:
    :return: sorted streams, the input is not modified
    """
    result = streams.copy()

    def stream_sort_key(stream: dict):
        codec_type = stream[K_CODEC_TYPE]
        if codec_type == CODEC_VIDEO:
            if stream.get(K_DISPOSITION, {}).get('attached_pic', 0) == 1:
                return 3
            return 0
        elif codec_type == CODEC_AUDIO:
            return 1
        else:
            return 2

    result.sort(key=stream_sort_key)
    return result


def get_media_title_from_tags(input_info: dict) -> [None, str]:
    if input_info is None:
        return None
    return input_info.get(K_FORMAT, {}).get(K_TAGS, {}).get(K_MEDIA_TITLE)


def get_media_title_from_filename(input_info: dict) -> [None, str]:
    if input_info is None:
        return None
    filename = input_info.get(K_FORMAT, {}).get("filename")
    if not filename:
        return None
    base_filename = os.path.basename(filename)
    return ".".join(base_filename.split('.')[0:-1])


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
    # TODO: adjust bitrates from recordings
    # TODO: adjust h265 crf to match desired qp
    qp = 28
    if target_height <= 480:
        if target_video_codec == 'h264':
            crf = 23
            bitrate = 1200
        else:
            crf = 28
            bitrate = 840
    elif target_height <= 720:
        if target_video_codec == 'h264':
            crf = 23
            bitrate = 2500
        else:
            crf = 28
            bitrate = 1750
    elif target_height <= 1080:
        # q=36 observed with libx264, crf=31
        qp = 34
        if target_video_codec == 'h264':
            crf = 31
            bitrate = 3500
        else:
            crf = 28
            bitrate = 2450
    else:
        # q=36 observed with libx264, crf=31
        qp = 34
        if target_video_codec == 'h264':
            crf = 31
            bitrate = 6500
        else:
            crf = 28
            bitrate = 4550

    return crf, bitrate, qp


def fps_video_filter(desired_frame_rate: [None, str, float]):
    if desired_frame_rate is None:
        return None
    return f"minterpolate=fps={desired_frame_rate}:mi_mode=blend"


def extend_opus_arguments(arguments, audio_info, current_output_stream, audio_filters=None, force_stereo=False):
    if audio_filters is None:
        audio_filters = []

    arguments.extend([f"-vbr:{current_output_stream}", "on"])

    # enables optimizations in opus v1.1+
    if force_stereo or audio_info.get('channel_layout') in ['mono', 'stereo', '1.0', '2.0']:
        arguments.extend([f"-mapping_family:{current_output_stream}", "0"])
    else:
        arguments.extend([f"-mapping_family:{current_output_stream}", "1"])

    if audio_info.get('channel_layout') == '5.1(side)':
        audio_filters.insert(0, "channelmap=channel_layout=5.1")
    elif audio_info.get('channel_layout') == '7.1(side)':
        audio_filters.insert(0, "channelmap=channel_layout=7.1")
    elif audio_info.get('channel_layout') == '4.0' and not force_stereo:
        # use ffmpeg upmix system to use more common 5.1 layout
        arguments.extend([f"-ac:{current_output_stream}", "6"])

    if audio_filters:
        arguments.extend([f"-filter:{current_output_stream}", ",".join(audio_filters)])

    # use original bit rate if lower than default
    target_bitrate = None
    channels = audio_info[K_CHANNELS]
    audio_bitrate = int(audio_info[K_BIT_RATE]) if K_BIT_RATE in audio_info else None
    if audio_bitrate is not None:
        if channels == 1 and audio_bitrate < (64 * 1024):
            target_bitrate = max(6, audio_bitrate)
        elif channels == 2 and audio_bitrate < (96 * 1024):
            target_bitrate = max(6, audio_bitrate)
        elif channels == 6 and audio_bitrate < (320 * 1024):
            target_bitrate = max(6, audio_bitrate)

    if target_bitrate is not None:
        arguments.extend([f"-b:{current_output_stream}", str(target_bitrate)])


def array_as_command(a) -> str:
    command = []
    for e in a:
        if '\'' in e:
            command.append('"' + e + '"')
        else:
            command.append("'" + e + "'")
    return ' '.join(command)


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
            parts = edl_line.split(maxsplit=3)
            if len(parts) < 3:
                continue
            start = parse_edl_ts(parts[0])
            end = parse_edl_ts(parts[1])
            if parts[2] == '0':
                event_type = EdlType.CUT
            elif parts[2] == '1':
                event_type = EdlType.MUTE
            elif parts[2] == '2':
                event_type = EdlType.SCENE
            elif parts[2] == '3':
                event_type = EdlType.COMMERCIAL
            elif parts[2] == '4':
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
        return round(float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2]), 3)
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
    with open(filepath, "w") as f:
        f.write(f"<Tags><Tag>\n")
        for k, v in tags.items():
            f.write(f"<Simple>"
                    f"<Name>{k}</Name>"
                    f"<String>{v}</String>"
                    f"</Simple>\n")
        f.write(f"</Tag></Tags>\n")


def parse_bytes(s: str) -> int:
    if s is None:
        return 0
    s = s.strip().lower()
    if len(s) == 0:
        return 0
    if s.endswith("b"):
        multiplier = 1
    elif s.endswith("k"):
        multiplier = KILOBYTES_MULT
    elif s.endswith("m"):
        multiplier = MEGABYTES_MULT
    elif s.endswith("g"):
        multiplier = GIGABYTES_MULT
    else:
        return int(s)
    return int(float(s[0:-1]) * multiplier)


def bytes_to_human_str(byte_count: int) -> str:
    if byte_count > GIGABYTES_MULT:
        return "{:.2f}G".format(float(byte_count) / GIGABYTES_MULT)
    if byte_count > MEGABYTES_MULT:
        return "{:.2f}M".format(float(byte_count) / MEGABYTES_MULT)
    if byte_count > KILOBYTES_MULT:
        return "{:.2f}M".format(float(byte_count) / KILOBYTES_MULT)
    return str(byte_count)


def parse_seconds(s: str) -> int:
    if s is None:
        return 0
    s = s.strip().lower()
    if len(s) == 0:
        return 0
    if s.endswith("s"):
        multiplier = 1
    elif s.endswith("m"):
        multiplier = 60
    elif s.endswith("h"):
        multiplier = 3600
    else:
        return int(s)
    return int(float(s[0:-1]) * multiplier)


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
        except (PermissionError, AccessDenied, ProcessLookupError):
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

    ffprobe = find_ffprobe()
    try:
        input_info = json.loads(subprocess.check_output(
            [ffprobe, '-v', 'quiet', '-analyzeduration', ANALYZE_DURATION, '-probesize', PROBE_SIZE, '-print_format',
             'json', '-show_format', '-show_streams', filepath]))
        return len(list(filter(lambda s: s[K_CODEC_TYPE] == CODEC_VIDEO, input_info.get('streams', [])))) > 0 and len(
            list(filter(lambda s: s[K_CODEC_TYPE] == CODEC_AUDIO, input_info.get('streams', [])))) > 0
    except subprocess.CalledProcessError:
        return False


def generate_video_files(args, suffix=".mkv"):
    """
    Searches for mkv files from the arguments that can be files and directories. A special case is two arguments, a
    mkv file and a missing second argument specifying the output file.
    :param args: list of files and/or directories
    :param suffix: the file suffix, default ".mkv"
    :return: yields a tuple of (input file, output file)
    """
    if len(args) == 2:
        # check special case of input file and output file
        if os.path.isfile(args[0]) and (os.path.isfile(args[1]) or not os.path.exists(args[1])):
            yield args[0], args[1]
            return

    for arg in args:
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
    chapters = input_info.get(K_CHAPTERS, [])
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
    if has_chapters_from_source_media(input_info)[0]:
        return False

    duration = float(input_info[K_FORMAT]['duration'])
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
    duration = float(input_info[K_FORMAT]['duration'])
    episode_duration = duration
    episode_count = 1
    filename = input_info[K_FORMAT]['filename']
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


def get_crop_filter_parts(crop_filter):
    """
    Split the parts of the crop filter into integers.
    :param crop_filter: similar to crop=100:100:20:8
    :return: ints [width, height, x, y]
    """
    if crop_filter is None:
        return None
    return [int(i) for i in crop_filter.split('=')[1].split(':')]


def _get_config_sources(filename: str):
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    # TODO: check Windows places
    return [f"{os.environ['HOME']}/.{filename}",
            f"/etc/{filename}",
            f"{script_dir}/{filename}"]


def find_config(filename: str) -> str:
    """
    Locate a config file from a list of common locations. The first one found is returned.
    See _get_config_sources for locations.
    :param filename: filename, like 'config.ini'
    :return: resolved path to a file that exists
    """
    sources = _get_config_sources(filename)
    for f in sources:
        if os.access(f, os.R_OK):
            return f
    raise OSError(f"Cannot find {filename} in any of {','.join(sources)}")


def load_config(filename: str, start_path=None, input_info=None,
                config: configparser.ConfigParser = None) -> configparser.ConfigParser:
    """
    Load configuration from one or more INI files. Overrides to the common configuration can be made by creating file(s)
    named :filename: in the directory tree of :start_path: or location of :input_info:.

    :param filename: the name of the config file, usually something like 'config.ini'
    :param start_path: the lowest directory or file of the directory tree to search for overrides
    :param input_info: if specified and the info contains the filename, set start_path from this
    :param config: an existing config to update
    :return: ConfigParser object
    """
    filenames = [find_config(filename)]

    if not start_path and input_info:
        input_path = input_info.get(K_FORMAT, {}).get("filename")
        if input_path:
            if os.path.isfile(input_path):
                start_path = os.path.dirname(input_path)
            elif os.path.isdir(input_path):
                start_path = input_path

    if start_path:
        path = start_path
        insert_idx = len(filenames)
        while True:
            p = os.path.dirname(os.path.abspath(path))
            if not p or p == path:
                break
            conf = os.path.join(p, filename)
            if os.path.isfile(conf):
                filenames.insert(insert_idx, conf)
            path = p

    if not config:
        config = configparser.ConfigParser()
    config.read(filenames)

    return config


config_obj: [configparser.ConfigParser, None] = None
_UNSET = object()


def get_global_config_option(section: str, option: str, fallback: [None, str] = _UNSET):
    """
    Get an option from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not _UNSET:
        return get_global_config().get(section, option, fallback=fallback)
    return get_global_config().get(section, option)


def get_global_config_boolean(section: str, option: str, fallback: bool = None):
    """
    Get a boolean from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not None:
        return get_global_config().getboolean(section, option, fallback=fallback)
    return get_global_config().getboolean(section, option)


def get_global_config_int(section: str, option: str, fallback: int = None):
    """
    Get an int from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not None:
        return get_global_config().getint(section, option, fallback=fallback)
    return get_global_config().getint(section, option)


def get_global_config_time_seconds(section: str, option: str, fallback: int = None):
    """
    Get number of seconds from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not None:
        return parse_seconds(get_global_config().get(section, option, fallback=fallback))
    return parse_seconds(get_global_config().get(section, option))


def get_global_config_bytes(section: str, option: str, fallback: int = None):
    """
    Get number of bytes from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not None:
        return parse_bytes(get_global_config().get(section, option, fallback=fallback))
    return parse_bytes(get_global_config().get(section, option))


def get_global_config_frame_rate(section: str, option: str, fallback: [None, str] = _UNSET) -> [None, str]:
    """
    Get the frame rate from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return: numeric frame rate, named frame rates are converted to numeric
    """

    if fallback == _UNSET:
        value = get_global_config().get(section, option)
    else:
        value = get_global_config().get(section, option, fallback=fallback)
    if value is None:
        return None
    return FRAME_RATE_NAMES.get(value.lower(), value)


def get_work_dir() -> str:
    return get_global_config_option('general', 'work_dir', fallback=tempfile.gettempdir())


def get_global_config() -> configparser.ConfigParser:
    global config_obj
    if config_obj is None:
        _config_lock.acquire()
        try:
            if config_obj is None:
                config_obj = _load_media_hare_config()
        finally:
            _config_lock.release()
    return config_obj


def _load_media_hare_config() -> configparser.ConfigParser:
    defaults = load_config('media-hare.defaults.ini')
    return load_config('media-hare.ini', config=defaults)


def match_owner_and_perm(target_path: str, source_path: str) -> bool:
    result = True
    source_stat = os.stat(source_path)
    try:
        os.chown(target_path, source_stat.st_uid, source_stat.st_gid)
    except OSError:
        logger.warning(f"Changing ownership of {target_path} failed, continuing")
        result = False

    try:
        os.chmod(target_path, source_stat.st_mode)
    except OSError:
        logger.warning(f"Changing permission of {target_path} failed, continuing")
        result = False

    return result


def is_truthy(value) -> bool:
    if value is None:
        return False
    return str(value).lower() in ['true', 't', 'yes', 'y', '1']


def frame_rate_from_s(frame_rate_s: [str, None]) -> [float, None]:
    if frame_rate_s is None:
        return None
    framerate: [float, None] = None
    frame_rate_s = frame_rate_s.lower()
    if frame_rate_s[0].isdigit():
        # remove suffix, like 'p'
        framerate = float(re.sub(r'\D', '', frame_rate_s))
    elif frame_rate_s in FRAME_RATE_NAMES.keys():
        framerate = float(eval(FRAME_RATE_NAMES[frame_rate_s]))
    else:
        logger.warning("Unknown framerate %s", frame_rate_s)
    return framerate


def should_adjust_frame_rate(current_frame_rate: [None, str, float], desired_frame_rate: [None, str, float],
                             tolerance: float = 0.25) -> bool:
    if current_frame_rate in [None, ''] or desired_frame_rate in [None, '']:
        return False

    if type(current_frame_rate) == str:
        current_frame_rate_f = eval(FRAME_RATE_NAMES.get(current_frame_rate.lower(), current_frame_rate))
    else:
        current_frame_rate_f = current_frame_rate

    if type(desired_frame_rate) == str:
        desired_frame_rate_f = eval(FRAME_RATE_NAMES.get(desired_frame_rate.lower(), desired_frame_rate))
    else:
        desired_frame_rate_f = desired_frame_rate

    frame_rate_pct = abs(current_frame_rate_f - desired_frame_rate_f) / max(current_frame_rate_f, desired_frame_rate_f)
    return frame_rate_pct >= tolerance
