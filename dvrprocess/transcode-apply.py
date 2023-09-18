#!/usr/bin/env python3

import getopt
import logging
import os
import sys
from subprocess import CalledProcessError

import requests

import common
from common import config, constants, hwaccel
from dvr_post_process import dvr_post_process
from find_need_transcode import need_transcode_generator

logger = logging.getLogger(__name__)


def usage():
    video_codecs = config.get_global_config_option('video', 'codecs')
    audio_codecs = config.get_global_config_option('audio', 'codecs')
    print(f"""{sys.argv[0]} [options] [media_paths]

Transcode content not matching desired video or audio codecs. This program is sensitive to current compute usage and
will not start running on a system under load and also will stop if the system becomes too loaded.

--dry-run
--verbose
-u, --url=
    Find files to process from a Plex Media Server. Specify the URL such as http://127.0.0.1:32400, default is {common.get_plex_url()}
-v, --video=
    Desired video codecs. Defaults to {video_codecs}
-a, --audio=
    Desired audio codecs. Defaults to only querying for video to transcode. Configured audio codecs are {audio_codecs}
-f, --framerate={','.join(constants.FRAME_RATE_NAMES.keys())},24,30000/1001,...
    Desired frame rate. If the current frame rate is within 25%, the file isn't considered.
--maxres=480
    Limit to specified height. Use to keep a lower powered machine from processing HD videos.
--ignore-compute
    Ignore current compute availability.
--limit=10
    Limit the number of files transcoded.
""", file=sys.stderr)


def transcode_apply(plex_url, media_paths=None, dry_run=False, desired_video_codecs=None, desired_audio_codecs=None,
                    desired_frame_rate=None, max_resolution=None,
                    verbose=False, limit: [None, int] = None):
    # collect libraries to scan in case filename changes, i.e. .ts to .mkv.
    libraries_to_scan = set()
    for file_info in need_transcode_generator(plex_url=plex_url, media_paths=media_paths,
                                              desired_video_codecs=desired_video_codecs,
                                              desired_audio_codecs=desired_audio_codecs, max_resolution=max_resolution,
                                              desired_frame_rate=desired_frame_rate):
        if limit is not None:
            limit -= 1
            if limit < 0:
                break

        original_size = -1
        try:
            original_size = os.stat(file_info.host_file_path).st_size
            post_process_code = dvr_post_process(file_info.host_file_path, dry_run=dry_run, verbose=verbose,
                                                 profanity_filter=True)
        except CalledProcessError as e:
            post_process_code = e.returncode
        except FileExistsError:
            # conflict with multiple processors
            post_process_code = 255
        if post_process_code == 0 and plex_url and file_info.item_key:
            file_changed = not os.path.exists(file_info.host_file_path) or os.stat(
                file_info.host_file_path).st_size != original_size
            if file_info.library and not os.path.exists(file_info.host_file_path):
                # We changed output types, i.e. filenames, Plex requires a library scan to find it
                libraries_to_scan.add(file_info.library)
            elif file_changed:
                # Updating a file without changing the name, we can analyze the item
                analyze_url = f'{plex_url}{file_info.item_key}/analyze'
                logger.info('HTTP PUT: %s', analyze_url)
                if not dry_run:
                    try:
                        requests.put(analyze_url)
                    except requests.exceptions.ConnectTimeout:
                        pass

    if plex_url and len(libraries_to_scan) > 0:
        for library_key in libraries_to_scan:
            refresh_url = f'{plex_url}/library/sections/{library_key}/refresh'
            logger.info('HTTP GET: %s', refresh_url)
            if not dry_run:
                try:
                    requests.get(refresh_url)
                except requests.exceptions.ConnectTimeout:
                    pass

    return 0


def should_transcode_run():
    if common.core_count() < 4 and hwaccel.find_hwaccel_method() == hwaccel.HWAccelMethod.NONE:
        return False
    return common.should_start_processing()


def transcode_apply_cli(argv):
    plex_url = common.get_plex_url()
    media_paths = None
    desired_video_codecs = None
    desired_audio_codecs = None
    desired_frame_rate = None
    max_resolution = None
    dry_run = False
    check_compute = True
    verbose = False
    limit = None

    try:
        opts, args = getopt.getopt(list(argv),
                                   "hnu:v:a:f:",
                                   ["dry-run", "ignore-compute", "url=", "video=", "audio=", "maxres=", "framerate=",
                                    "verbose", "limit="])
    except getopt.GetoptError:
        usage()
        return 2
    for opt, arg in opts:
        if opt == '-h':
            usage()
            return 2
        elif opt in ("-n", "--dry-run"):
            dry_run = True
        elif opt in ("-u", "--url"):
            plex_url = arg
        elif opt in ("-v", "--video"):
            desired_video_codecs = arg.split(',')
        elif opt in ("-a", "--audio"):
            desired_audio_codecs = arg.split(',')
        elif opt in ("-f", "--framerate"):
            desired_frame_rate = arg
        elif opt == '--maxres':
            max_resolution = int(arg)
        elif opt == '--verbose':
            verbose = True
            logging.root.setLevel(logging.DEBUG)
        elif opt == '--ignore-compute':
            check_compute = False
        elif opt == '--limit':
            limit = int(arg)

    if args:
        media_paths = args
        plex_url = None
    elif not plex_url:
        media_paths = common.get_media_paths()

    if common.check_already_running():
        return 0

    if check_compute and not should_transcode_run():
        logger.warning(f"not enough compute available for transcoding")
        return 255

    transcode_apply(plex_url, media_paths=media_paths, dry_run=dry_run, desired_video_codecs=desired_video_codecs,
                    desired_audio_codecs=desired_audio_codecs, desired_frame_rate=desired_frame_rate,
                    max_resolution=max_resolution, verbose=verbose, limit=limit)
    return 0


if __name__ == '__main__':
    common.cli_wrapper(transcode_apply_cli)
