#!/usr/bin/env python3

import getopt
import logging
import os
import random
import sys
import xml.etree.ElementTree as ET

import requests

import common

#
# Developer notes:
# List of libraries: /library/sections, want type="show"
# Shows for a section: /library/sections/2/all from Directory.key in section
# All Episodes: /library/metadata/83179/allLeaves from Directory.key in show
#

# TODO: These time multipliers are very specific to my hardware. They should be based on the number of cores,
#  configured preset, codec, and will probably be widely inaccurate :p
TRANSCODE_MULTIPLER_480 = 2.0
TRANSCODE_MULTIPLER_720 = 0.8
TRANSCODE_MULTIPLER_1080 = 0.5
TRANSCODE_MULTIPLER_4K = 0.2

logger = logging.getLogger(__name__)


def usage():
    video_codecs = common.get_global_config_option('video', 'codecs')
    audio_codecs = common.get_global_config_option('audio', 'codecs')
    print(f"""{sys.argv[0]}

List files needing transcode from the Plex database. There are two output formats:

1. Absolute paths terminated with null (this can be changed) with the intent to be piped into xargs or similar tool.
   If you intend to use this file list to transcode, see transcode-apply.py instead.
2. Nagios monitoring output, which is also human readable. This also provides some estimates on time to transcode.

-u, --url=
    Find files to process from a Plex Media Server. Specify the URL such as http://127.0.0.1:32400, default is {common.get_plex_url()}
-t, --terminator="\\n"
    Set the output terminator, defaults to null (0).
-d, --dir=
    Directory containing media. Defaults to {common.get_media_base()}
-v, --video=
    Desired video codecs. Defaults to {video_codecs}
-a, --audio=
    Desired audio codecs. Defaults to only querying for video to transcode. Configured audio codecs are {audio_codecs}
--maxres=480
    Limit to specified height. Use to keep a lower powered machine from processing HD videos.
--nagios
    Output for Nagios monitoring. Also human readable with statistics and estimates of transcode time.
""", file=sys.stderr)


def find_need_transcode_cli(argv):
    plex_url = common.get_plex_url()
    host_home = common.get_media_base()
    terminator = '\0'
    desired_video_codecs = None
    desired_audio_codecs = None
    nagios_output = False
    max_resolution = None

    try:
        opts, args = getopt.getopt(argv, "hu:t:d:v:a:",
                                   ["url=", "terminator=", "dir=", "video=", "audio=", "nagios", "maxres="])
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
            host_home = arg
        elif opt in ("-v", "--video"):
            desired_video_codecs = arg.split(',')
        elif opt in ("-a", "--audio"):
            desired_audio_codecs = arg.split(',')
        elif opt == '--nagios':
            nagios_output = True
        elif opt == '--maxres':
            max_resolution = int(arg)
        elif opt in ("-t", "--terminator"):
            if arg == '\\n':
                terminator = '\n'
            elif arg == '\\0':
                terminator = '\0'
            else:
                terminator = arg

    if nagios_output:
        file_names = set()
        runtime_minutes = 0
        transcode_minutes = 0
        transcode_details = []
        for file_info in need_transcode_generator(plex_url, host_home, desired_video_codecs, desired_audio_codecs,
                                                  max_resolution):
            file_names.update([file_info.file_name])
            runtime_minutes += file_info.runtime
            transcode_minutes += file_info.transcode_time
            transcode_details.extend(
                [f"{file_info.file_name};{file_info.video_resolution};{file_info.transcode_time};{file_info.runtime}"])

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
        for file_info in need_transcode_generator(plex_url, host_home, desired_video_codecs, desired_audio_codecs,
                                                  max_resolution):
            sys.stdout.write(file_info.file_name)
            sys.stdout.write(terminator)
        return 0


class TranscodeFileInfo(object):

    def __init__(self, file_name, host_file_path, item_key, video_resolution, transcode_time, runtime):
        self.file_name = file_name
        self.host_file_path = host_file_path
        self.item_key = item_key
        self.video_resolution = video_resolution
        self.transcode_time = transcode_time
        self.runtime = runtime


def need_transcode_generator(
        plex_url=common.get_plex_url(),
        host_home=None,
        desired_video_codecs: list[str] = None,
        desired_audio_codecs: list[str] = None,
        max_resolution=None
):
    if not plex_url:
        raise Exception("No plex URL, configure in media-hare.ini, section plex, option url")
    if desired_video_codecs is None and desired_audio_codecs is None:
        desired_video_codecs = common.get_global_config_option('video', 'codecs')
    if host_home is None:
        host_home = common.get_media_base()

    file_names = set()

    sections_response = requests.get(f'{plex_url}/library/sections')
    sections = list(ET.fromstring(sections_response.text))
    random.shuffle(sections)
    for library in sections:
        if library.tag == 'Directory' and library.attrib['type'] == 'movie':
            section_response = requests.get(
                f'{plex_url}/library/sections/{library.attrib["key"]}/all')
            yield from _process_videos(desired_audio_codecs, desired_video_codecs, file_names, host_home,
                                       section_response,
                                       max_resolution)
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
                yield from _process_videos(desired_audio_codecs, desired_video_codecs, file_names, host_home,
                                           show_response,
                                           max_resolution)


def _process_videos(desired_audio_codecs: list[str], desired_video_codecs: list[str], file_names, host_home,
                    show_response, max_resolution):
    episodes = list(filter(
        lambda el: el.tag == 'Video' and (
                el.attrib['type'] == 'episode' or el.attrib['type'] == 'movie'),
        ET.fromstring(show_response.text)))
    random.shuffle(episodes)
    for episode in episodes:
        video_codec = "?"
        video_resolution = None
        audio_codec = "?"
        file_name = "?"
        file_size = 0
        duration = episode.attrib.get("duration")
        if duration is not None:
            duration = int(int(duration) / 60000)
        for media in list(filter(lambda el: el.tag == 'Media', episode)):
            video_codec = common.resolve_human_codec(media.attrib.get("videoCodec", "?"))
            audio_codec = common.resolve_human_codec(media.attrib.get("audioCodec", "?"))
            video_resolution = media.attrib.get("videoResolution")
            if video_resolution == 'sd':
                video_resolution = '480'
            for part in list(filter(lambda el: el.tag == 'Part', media)):
                file_name = part.attrib.get("file", "?")
                file_size = int(part.attrib.get("size", "0"))

        if ((desired_video_codecs is not None and video_codec not in desired_video_codecs) or (
                desired_audio_codecs is not None and audio_codec not in desired_audio_codecs)) and file_name != '?' \
                and (
                max_resolution is None or (video_resolution is not None and int(video_resolution) <= max_resolution)):

            # Verify file hasn't changed
            host_file_path, file_name = _plex_host_name_to_local(file_name, host_home)
            if host_file_path is None:
                continue

            if os.path.isfile(host_file_path):
                current_file_size = os.path.getsize(host_file_path)
                if current_file_size == file_size and file_name not in file_names:
                    file_names.update([file_name])
                    if video_resolution is not None:
                        if video_resolution.endswith("k"):
                            video_resolution = float(video_resolution[0:-1]) * 1000
                        else:
                            video_resolution = int(video_resolution)
                        if video_resolution <= 480:
                            multiplier = TRANSCODE_MULTIPLER_480
                        elif video_resolution <= 720:
                            multiplier = TRANSCODE_MULTIPLER_720
                        elif video_resolution <= 1080:
                            multiplier = TRANSCODE_MULTIPLER_1080
                        else:
                            multiplier = TRANSCODE_MULTIPLER_4K
                        transcode_this_file = int(duration / multiplier)
                        yield TranscodeFileInfo(file_name, host_file_path, episode.attrib['key'], video_resolution,
                                                transcode_this_file,
                                                duration)


def _plex_host_name_to_local(file_name: str, host_home: str) -> (str, str):
    """
    Try to find a file referenced by the Plex host name to the local host name.
    :param file_name: the file name on the host running Plex
    :param host_home: the current host media home
    :return: valid file name or None
    """
    paths = common.get_global_config_option('media', 'paths').split(',')
    for path in paths:
        i = file_name.find(path)
        if i >= 0:
            f = os.path.join(host_home, file_name[i:])
            logger.debug(f"Checking if %s is a valid path for %s", f, file_name)
            if os.path.isfile(f):
                return f, file_name[i:]
    return None, None


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(find_need_transcode_cli(sys.argv[1:]))
