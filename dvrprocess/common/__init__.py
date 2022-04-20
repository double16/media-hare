import _thread
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
from enum import Enum
from shutil import which

import psutil
from psutil import AccessDenied

KILOBYTES_MULT = 1024
MEGABYTES_MULT = 1024 * 1024
GIGABYTES_MULT = 1024 * 1024 * 1024

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

K_STREAM_TITLE = 'title'
K_STREAM_INDEX = 'index'
K_CODEC_NAME = 'codec_name'
K_CODEC_TYPE = 'codec_type'
K_FORMAT = 'format'
K_TAGS = 'tags'
K_CHAPTERS = 'chapters'
K_BIT_RATE = 'bit_rate'
K_CHANNELS = 'channels'
K_FILTER_VERSION = 'PFILTER_VERSION'
K_FILTER_HASH = 'PFILTER_HASH'
K_FILTER_SKIP = 'PFILTER_SKIP'
K_FILTER_STOPPED = 'PFILTER_STOPPED'
K_COMSKIP_HASH = 'COMSKIP_HASH'

CODEC_SUBTITLE_ASS = 'ass'
CODEC_SUBTITLE_SRT = 'srt'
CODEC_SUBTITLE_SUBRIP = 'subrip'
CODEC_SUBTITLE_DVDSUB = 'dvd_subtitle'
CODEC_SUBTITLE_BLURAY = 'hdmv_pgs_subtitle'
LANGUAGE_ENGLISH = 'eng'
CODEC_SUBTITLE = 'subtitle'
CODEC_AUDIO = 'audio'
CODEC_VIDEO = 'video'

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


def get_media_paths():
    base = get_media_base()
    return [f"{base}/Movies",
            f"{base}/DVRShows",
            f"{base}/TVShows",
            f"{base}/ToReview/Movies",
            f"{base}/PG/Movies",
            f"{base}/PG/DVRShows",
            f"{base}/PG/TVShows",
            ]


def find_media_base():
    for p in psutil.disk_partitions(all=True):
        base = os.path.join(p.mountpoint, 'Media')
        if os.path.isdir(os.path.join(base, 'Movies')):
            return base
    host_home = '/var/lib/dropbox/docsdata/Media'
    if os.path.isdir(host_home):
        return host_home
    home = os.path.join(os.environ['HOME'], 'Dropbox', 'Media')
    if os.path.isdir(home):
        return home
    return "/home/Dropbox/Media"


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


ffmpeg_path = None


def find_ffmpeg():
    global ffmpeg_path
    if ffmpeg_path is None:
        _once_lock.acquire()
        try:
            if ffmpeg_path is None:
                ffmpeg_path = _find_ffmpeg()
        finally:
            _once_lock.release()
    return ffmpeg_path


def _find_ffmpeg():
    ffmpeg = "x /usr/lib/plexmediaserver/Plex\\ Transcoder"

    if os.access(ffmpeg, os.X_OK):
        ld_library_path = "/usr/lib/plexmediaserver:/usr/lib/plexmediaserver/lib"
    else:
        ffmpeg = which("ffmpeg")
    if len(ffmpeg) == 0:
        fatal("'/usr/lib/plexmediaserver/Plex Transcoder' nor 'ffmpeg' found")
    if not os.access(ffmpeg, os.X_OK):
        fatal(f"{ffmpeg} is not an executable")

    return ffmpeg


ffprobe_path = None


def find_ffprobe():
    global ffprobe_path
    if ffprobe_path is None:
        _once_lock.acquire()
        try:
            if ffprobe_path is None:
                ffprobe_path = _find_ffprobe()
        finally:
            _once_lock.release()
    return ffprobe_path


def _find_ffprobe():
    ffprobe = which("ffprobe")
    if not os.access(ffprobe, os.X_OK):
        fatal(f"{ffprobe} is not an executable")
    return ffprobe


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


def get_plex_url():
    return "http://192.168.1.254:32400"


def load_keyframes_by_seconds(filepath) -> list[float]:
    ffprobe_keyframes = [find_ffprobe(), "-loglevel", "error", "-skip_frame", "nokey", "-select_streams", "v:0",
                         "-show_entries", "frame=pkt_pts_time", "-of", "csv=print_section=0", filepath]
    keyframes = list(map(lambda s: float(s),
                         filter(lambda s: len(s) > 0,
                                subprocess.check_output(ffprobe_keyframes, universal_newlines=True,
                                                        text=True).splitlines())))
    return keyframes


def find_desired_keyframe(keyframes: list[float], start: float, force_after=False) -> float:
    if len(keyframes) == 0:
        return start
    elif len(keyframes) == 1:
        return keyframes[0]
    elif len(keyframes) == 2:
        if force_after:
            return keyframes[1]
        d1 = abs(keyframes[0] - start)
        d2 = abs(keyframes[1] - start)
        if d1 <= d2:
            return keyframes[0]
        else:
            return keyframes[1]
    mid = len(keyframes) // 2
    if start < keyframes[mid]:
        return find_desired_keyframe(keyframes[:mid + 1], start)
    else:
        return find_desired_keyframe(keyframes[mid:], start)


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


def find_video_stream(input_info):
    streams = list(filter(lambda stream: stream[K_CODEC_TYPE] == CODEC_VIDEO, input_info['streams']))
    english = find_english_streams(streams)
    if len(english) > 0:
        streams = english
    return streams[0]


def find_audio_streams(input_info):
    streams = list(filter(lambda stream: stream['codec_type'] == CODEC_AUDIO, input_info['streams']))

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
        filter(lambda stream: stream.get('disposition') and stream.get('disposition').get('default') > 0, streams))
    if len(default_streams) > 0:
        return default_streams[0:1]

    return streams


def resolve_video_codec(desired_codec, target_height=None, video_info=None):
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
        # my testing shows bitrate for h264 is lower than h265 and ~10x faster
        # if target_height and target_height > 1080:
        #    result = "h265"

    return result


def resolve_audio_codec(desired_codec, audio_info=None):
    """
    Resolve to a real audio codec as seen by the user (NOT the ffmpeg version). This may be adjusted based on the
    audio info.
    :param desired_codec:
    :param audio_info:
    :return:
    """
    return __resolve_codec(desired_codec, audio_info)


def __resolve_codec(desired_codec, stream_info=None):
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
        if result is None:
            for idx, codec in enumerate(desired_codec):
                result = codec
                break
    else:
        result = desired_codec

    return resolve_human_codec(result)


def resolve_human_codec(codec):
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


def ffmpeg_codec(desired_codec):
    if re.search("h[0-9][0-9][0-9]", desired_codec):
        return f"libx{desired_codec[1:]}"
    if desired_codec == "opus":
        return "libopus"
    return desired_codec


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
    qp = 22
    if target_height <= 480:
        if target_video_codec == 'h264':
            crf = 23
        else:
            crf = 28
        # h264
        bitrate = 1200
        # bitrate = 1400
    elif target_height <= 720:
        if target_video_codec == 'h264':
            crf = 23
        else:
            crf = 28
        # h264
        bitrate = 2500
    elif target_height <= 1080:
        if target_video_codec == 'h264':
            crf = 31
        else:
            crf = 28
        # Optimal for 1080p, h264
        bitrate = 3500
    else:
        if target_video_codec == 'h264':
            crf = 31
        else:
            crf = 28
        # Optimal for 4k, h264
        bitrate = 6500

    return crf, bitrate, qp


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


def array_as_command(a):
    return ' '.join(map(lambda e: f"'{e}'", a))


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


def parse_edl(filename) -> list[EdlEvent]:
    events = []
    with open(filename, 'r') as edl_fd:
        for edl_line in edl_fd.readlines():
            if edl_line.startswith('##'):
                continue
            parts = edl_line.split(maxsplit=3)
            if len(parts) < 3:
                continue
            start = __parse_edl_ts(parts[0])
            end = __parse_edl_ts(parts[1])
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
    return events


def parse_edl_cuts(filename) -> list[EdlEvent]:
    return list(filter(lambda e: e.event_type in [EdlType.CUT, EdlType.COMMERCIAL], parse_edl(filename)))


def __parse_edl_ts(s: str) -> float:
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
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


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
    logger.error('', exc_info=e)


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
