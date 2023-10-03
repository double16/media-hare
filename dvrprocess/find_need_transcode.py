#!/usr/bin/env python3

import getopt
import logging
import os
import random
import sys
from typing import Tuple, Union
import xml.etree.ElementTree as ET

import requests

import common
from common import config, constants

#
# Developer notes:
# List of libraries: /library/sections, want type="show"
# Shows for a section: /library/sections/2/all from Directory.key in section
# All Episodes: /library/metadata/83179/allLeaves from Directory.key in show
#

# TODO: These time multipliers are very specific to my hardware. They should be based on the number of cores,
#  configured preset, codec, and will probably be wildly inaccurate :p
TRANSCODE_MULTIPLER_480 = 2.0
TRANSCODE_MULTIPLER_720 = 0.8
TRANSCODE_MULTIPLER_1080 = 0.5
TRANSCODE_MULTIPLER_4K = 0.2

logger = logging.getLogger(__name__)


def usage():
    video_codecs = config.get_global_config_option('video', 'codecs')
    audio_codecs = config.get_global_config_option('audio', 'codecs')
    print(f"""{sys.argv[0]} [options] [media_paths]

List files needing transcode from the Plex database. There are two output formats:

1. Absolute paths terminated with null (this can be changed) with the intent to be piped into xargs or similar tool.
   If you intend to use this file list to transcode, see transcode-apply.py instead.
2. Nagios monitoring output, which is also human readable. This also provides some estimates on time to transcode.

-u, --url=
    Find files to process from a Plex Media Server. Specify the URL such as http://127.0.0.1:32400, default is {common.get_plex_url()}
-t, --terminator="\\n"
    Set the output terminator, defaults to null (0).
-d, --dir=
    Directory containing media. Defaults to {common.get_media_roots()}
-v, --video=
    Desired video codecs. Defaults to {video_codecs}
-a, --audio=
    Desired audio codecs. Defaults to only querying for video to transcode. Configured audio codecs are {audio_codecs}
-f, --framerate={','.join(constants.FRAME_RATE_NAMES.keys())},24,30000/1001,...
    Desired frame rate. If the current frame rate is within 25%, the file isn't considered.
--maxres=480
    Limit to specified height. Use to keep a lower powered machine from processing HD videos.
--nagios
    Output for Nagios monitoring. Also human readable with statistics and estimates of transcode time.
""", file=sys.stderr)


def find_need_transcode_cli(argv):
    plex_url = common.get_plex_url()
    media_paths = None
    roots = []
    terminator = '\0'
    desired_video_codecs = None
    desired_audio_codecs = None
    nagios_output = False
    max_resolution = None
    desired_frame_rate = None

    try:
        opts, args = getopt.getopt(argv, "hu:t:d:v:a:f:",
                                   ["url=", "terminator=", "dir=", "video=", "audio=", "nagios", "maxres=",
                                    "framerate="])
    except getopt.GetoptError:
        usage()
        return 2
    for opt, arg in opts:
        if opt == '-h':
            usage()
            return 2
        elif opt in ("-u", "--url"):
            plex_url = arg
        elif opt in ("-d", "--dir"):
            roots.append(arg)
        elif opt in ("-v", "--video"):
            desired_video_codecs = arg.split(',')
        elif opt in ("-a", "--audio"):
            desired_audio_codecs = arg.split(',')
        elif opt == '--nagios':
            nagios_output = True
        elif opt in ("-f", "--framerate"):
            desired_frame_rate = arg
        elif opt == '--maxres':
            max_resolution = int(arg)
        elif opt in ("-t", "--terminator"):
            if arg == '\\n':
                terminator = '\n'
            elif arg == '\\0':
                terminator = '\0'
            else:
                terminator = arg

    if not roots:
        roots = common.get_media_roots()

    if args:
        media_paths = args
        plex_url = None
    elif not plex_url:
        media_paths = common.get_media_paths()

    if nagios_output:
        file_names = set()
        runtime_minutes = 0
        transcode_minutes = 0
        transcode_details = []
        for file_info in need_transcode_generator(plex_url=plex_url, media_paths=media_paths, media_roots=roots,
                                                  desired_video_codecs=desired_video_codecs,
                                                  desired_audio_codecs=desired_audio_codecs,
                                                  max_resolution=max_resolution,
                                                  desired_frame_rate=desired_frame_rate):
            file_names.update([file_info.file_name])
            runtime_minutes += file_info.runtime
            transcode_minutes += file_info.transcode_time
            transcode_details.extend(
                [
                    f"{file_info.file_name};{file_info.video_resolution};{file_info.framerate};{file_info.transcode_time};{file_info.runtime}"])

        # one week
        if transcode_minutes > 10080:
            level = "CRITICAL"
            code = 2
        # two days
        elif transcode_minutes > 2880:
            level = "WARNING"
            code = 1
        else:
            level = "OK"
            code = 0

        print(
            f"TRANSCODE_PENDING {level}: {round(transcode_minutes / 60, 2)}h, runtime: {round(runtime_minutes / 60, 2)}h, files: {len(file_names)} | TRANSCODE_PENDING;{transcode_minutes};{runtime_minutes};{len(file_names)}")
        for e in transcode_details:
            print(e)
        return code
    else:
        for file_info in need_transcode_generator(plex_url=plex_url, media_paths=media_paths, media_roots=roots,
                                                  desired_video_codecs=desired_video_codecs,
                                                  desired_audio_codecs=desired_audio_codecs,
                                                  max_resolution=max_resolution,
                                                  desired_frame_rate=desired_frame_rate):
            sys.stdout.write(file_info.file_name)
            sys.stdout.write(terminator)
        return 0


class TranscodeFileInfo(object):

    def __init__(self, file_name: str, host_file_path: str, item_key: Union[str, None], video_resolution: int,
                 runtime: int,
                 framerate: Union[None, float] = None, library: Union[None, str] = None):
        self.file_name = file_name
        self.host_file_path = host_file_path
        self.item_key = item_key
        self.video_resolution = video_resolution
        self.runtime = runtime
        self.transcode_time = runtime
        self.framerate = framerate
        self.library = library

        if video_resolution is not None and runtime is not None:
            if video_resolution <= 480:
                multiplier = TRANSCODE_MULTIPLER_480
            elif video_resolution <= 720:
                multiplier = TRANSCODE_MULTIPLER_720
            elif video_resolution <= 1080:
                multiplier = TRANSCODE_MULTIPLER_1080
            else:
                multiplier = TRANSCODE_MULTIPLER_4K
            # TODO: base upon available compute, codec and preset
            self.transcode_time = int(runtime / multiplier)


def _os_walk_media_generator(media_paths, desired_audio_codecs: list[str], desired_video_codecs: list[str],
                             desired_subtitle_codecs: list[str],
                             max_resolution: Union[None, int], desired_frame_rate: Union[None, float]):
    random.shuffle(media_paths)
    for media_path in media_paths:
        for root, dirs, files in os.walk(media_path, topdown=True):
            random.shuffle(dirs)
            random.shuffle(files)

            for file in common.filter_for_mkv(files):
                filepath = os.path.join(root, file)
                input_info = common.find_input_info(filepath)
                if not input_info:
                    continue

                need_transcode = False

                video_streams = common.find_video_streams(input_info)

                min_height = min(map(lambda e: int(e['height']), video_streams))
                if max_resolution is not None and min_height > max_resolution:
                    continue

                min_framerate = min(map(lambda e: float(eval(e['avg_frame_rate'])), video_streams))
                if common.should_adjust_frame_rate(current_frame_rate=min_framerate,
                                                   desired_frame_rate=desired_frame_rate):
                    need_transcode = True

                if desired_video_codecs is not None:
                    video_codecs = set(map(lambda e: e[constants.K_CODEC_NAME], video_streams))
                    if not video_codecs.issubset(set(desired_video_codecs)):
                        need_transcode = True

                audio_streams = list(
                    filter(lambda stream: stream[constants.K_CODEC_TYPE] == constants.CODEC_AUDIO,
                           input_info['streams']))
                if desired_audio_codecs is not None:
                    audio_codecs = set(map(lambda e: e[constants.K_CODEC_NAME], audio_streams))
                    if not audio_codecs.issubset(set(desired_audio_codecs)):
                        need_transcode = True

                if desired_subtitle_codecs is not None:
                    subtitle_streams = list(
                        filter(lambda stream: stream[constants.K_CODEC_TYPE] == constants.CODEC_SUBTITLE,
                               input_info['streams']))
                    subtitle_codecs = set(map(lambda e: e[constants.K_CODEC_NAME], subtitle_streams))
                    if len(subtitle_codecs) == 0:
                        need_transcode = len(audio_streams) > 0
                    elif not subtitle_codecs.intersection(set(desired_subtitle_codecs)):
                        need_transcode = True

                if need_transcode:
                    yield TranscodeFileInfo(file_name=file, host_file_path=filepath, video_resolution=min_height,
                                            runtime=int(float(input_info[constants.K_FORMAT]['duration'])),
                                            framerate=min_framerate, item_key=None)


def need_transcode_generator(
        plex_url=common.get_plex_url(),
        media_paths=None,
        media_roots=None,
        desired_video_codecs: list[str] = None,
        desired_audio_codecs: list[str] = None,
        desired_subtitle_codecs: list[str] = None,
        max_resolution: Union[None, int] = None,
        desired_frame_rate: Union[None, float] = None,
):
    logger.debug(
        "need_transcode_generator(plex_url=%s, media_paths=%s, media_roots=%s, desired_video_codecs=%s, "
        "desired_audio_codecs=%s, desired_subtitle_codecs=%s, max_resolution=%s, desired_frame_rate=%s)",
        plex_url, media_paths, media_roots, desired_video_codecs, desired_audio_codecs, desired_subtitle_codecs,
        max_resolution, desired_frame_rate)

    if desired_video_codecs is None and desired_audio_codecs is None and desired_frame_rate is None:
        desired_video_codecs = config.get_global_config_option('video', 'codecs').split(',')
        desired_audio_codecs = config.get_global_config_option('audio', 'codecs').split(',')
        desired_frame_rate = config.get_global_config_frame_rate('post_process', 'frame_rate', None)

    if media_paths:
        yield from _os_walk_media_generator(media_paths, desired_video_codecs=desired_video_codecs,
                                            desired_audio_codecs=desired_audio_codecs,
                                            desired_subtitle_codecs=desired_subtitle_codecs,
                                            max_resolution=max_resolution,
                                            desired_frame_rate=desired_frame_rate)
        return
    if not plex_url:
        raise Exception("No plex URL, configure in media-hare.ini, section plex, option url")

    if not media_roots:
        media_roots = common.get_media_roots()

    file_names = set()

    sections_response = requests.get(f'{plex_url}/library/sections')
    sections = list(ET.fromstring(sections_response.text))
    random.shuffle(sections)
    for library in sections:
        library_key = library.attrib['key']
        if library.tag == 'Directory' and library.attrib['type'] == 'movie':
            section_response = requests.get(
                f'{plex_url}/library/sections/{library.attrib["key"]}/all')
            yield from _process_videos(desired_audio_codecs, desired_video_codecs, desired_subtitle_codecs, file_names,
                                       media_roots,
                                       section_response,
                                       max_resolution,
                                       desired_frame_rate,
                                       library_key)
        elif library.tag == 'Directory' and library.attrib['type'] == 'show' and 'DVR' not in library.attrib['title']:
            section_response = requests.get(
                f'{plex_url}/library/sections/{library.attrib["key"]}/all')
            shows = list(filter(lambda el: el.tag ==
                                           'Directory' and el.attrib['type'] == 'show',
                                ET.fromstring(section_response.text)))
            random.shuffle(shows)
            for show in shows:
                show_response = requests.get(
                    f'{plex_url}{show.attrib["key"].replace("/children", "/allLeaves")}')
                yield from _process_videos(desired_audio_codecs, desired_video_codecs, desired_subtitle_codecs,
                                           file_names, media_roots,
                                           show_response,
                                           max_resolution,
                                           desired_frame_rate,
                                           library_key)


def _process_videos(desired_audio_codecs: list[str], desired_video_codecs: list[str],
                    desired_subtitle_codecs: list[str], file_names, media_roots: list[str],
                    show_response, max_resolution: Union[None, int], desired_frame_rate: Union[None, float],
                    library: Union[None, str] = None):
    episodes = list(filter(
        lambda el: el.tag == 'Video' and (
                el.attrib['type'] == 'episode' or el.attrib['type'] == 'movie'),
        ET.fromstring(show_response.text)))
    random.shuffle(episodes)
    for episode in episodes:
        video_codec = "?"
        video_resolution = None
        framerate = None
        audio_codec = "?"
        subtitle_codec = None
        file_name = "?"
        file_size = 0
        duration = episode.attrib.get("duration")
        if duration is not None:
            duration = int(int(duration) / 60000)
        for media in list(filter(lambda el: el.tag == 'Media', episode)):
            video_codec = common.resolve_human_codec(media.attrib.get("videoCodec", "?"))
            audio_codec = common.resolve_human_codec(media.attrib.get("audioCodec", "?"))
            video_resolution = media.attrib.get("videoResolution")
            framerate = common.frame_rate_from_s(media.attrib.get("videoFrameRate"))

            if video_resolution == 'sd':
                video_resolution = '480'
            elif video_resolution == '4k':
                video_resolution = '2160'
            elif video_resolution == '8k':
                video_resolution = '4320'

            for part in list(filter(lambda el: el.tag == 'Part', media)):
                file_name = part.attrib.get("file", "?")
                file_size = int(part.attrib.get("size", "0"))
                # TODO: stream info isn't coming back with this call, move to python-plexapi
                # <Stream id="347880" streamType="3" default="1" codec="ass" index="2" bitrate="0" language="English" languageTag="en" languageCode="eng" title="Filtered" displayTitle="English (ASS)" extendedDisplayTitle="Filtered (English ASS)">
                streams = list(filter(lambda el: el.tag == 'Stream', part))
                if len(streams) == 0:
                    subtitle_codec = "?"
                for subtitle_stream in list(
                        filter(lambda el: el.tag == 'Stream' and el.attrib.get('streamType', "") == "3", part)):
                    if desired_subtitle_codecs is not None and subtitle_stream.attrib.get(
                            'codec') in desired_subtitle_codecs:
                        subtitle_codec = subtitle_stream.attrib.get('codec')
                    elif subtitle_codec is None:
                        subtitle_codec = subtitle_stream.attrib.get('codec')

        # if we want subtitles but there is no subtitle stream of any kind, there is no way to transcode, so skip it
        if desired_subtitle_codecs is not None and audio_codec in [None, "?"]:
            logger.info("%s: Skipping because there are no subtitle nor audio streams", file_name)
            continue

        # if we want specific audio codecs but there is no audio stream, there is no way to transcode, so skip it
        if desired_audio_codecs is not None and audio_codec == "?":
            continue

        if ((desired_video_codecs is not None and video_codec not in desired_video_codecs) or (
                desired_audio_codecs is not None and audio_codec not in desired_audio_codecs) or (
                    desired_subtitle_codecs is not None and subtitle_codec not in desired_subtitle_codecs) or (
                    common.should_adjust_frame_rate(current_frame_rate=framerate,
                                                    desired_frame_rate=desired_frame_rate))) \
                and file_name != '?' \
                and (
                max_resolution is None or (video_resolution is not None and int(video_resolution) <= max_resolution)):

            # Verify file hasn't changed
            host_file_path, file_name = _plex_host_name_to_local(file_name, media_roots)
            if host_file_path is None:
                continue

            if os.path.isfile(host_file_path):
                current_file_size = os.path.getsize(host_file_path)
                if current_file_size == file_size and file_name not in file_names:
                    file_names.update([file_name])
                    if video_resolution is not None:
                        if video_resolution.endswith("k"):
                            video_resolution = int(float(video_resolution[0:-1]) * 1000)
                        else:
                            video_resolution = int(video_resolution)
                        yield TranscodeFileInfo(file_name=file_name, host_file_path=host_file_path,
                                                item_key=episode.attrib['key'], video_resolution=video_resolution,
                                                framerate=framerate, runtime=duration, library=library)


def _plex_host_name_to_local(file_name: str, media_roots: list[str]) -> Tuple[str, str]:
    """
    Try to find a file referenced by the Plex host name to the local host name.
    :param file_name: the file name on the host running Plex
    :param media_roots: roots where media is stored
    :return: valid file name or None
    """
    paths = config.get_global_config_option('media', 'paths').split(',')
    for path in paths:
        i = file_name.find(path)
        if i >= 0:
            for root in media_roots:
                f = os.path.join(root, file_name[i:])
                logger.debug(f"Checking if %s is a valid path for %s", f, file_name)
                if os.path.isfile(f):
                    return f, file_name[i:]
    return None, None


if __name__ == '__main__':
    common.setup_cli(level=logging.ERROR, start_gauges=False)
    sys.exit(find_need_transcode_cli(sys.argv[1:]))
