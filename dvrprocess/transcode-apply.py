#!/usr/bin/env python3

import getopt
import logging
import sys
from subprocess import CalledProcessError

import requests

import common
from dvr_post_process import dvr_post_process
from find_need_transcode import need_transcode_generator

logger = logging.getLogger(__name__)


def usage():
    video_codecs = common.get_global_config_option('video', 'codecs')
    audio_codecs = common.get_global_config_option('audio', 'codecs')
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
--maxres=480
    Limit to specified height. Use to keep a lower powered machine from processing HD videos.
--ignore-compute
    Ignore current compute availability.
""", file=sys.stderr)


def transcode_apply(plex_url, media_paths=None, dry_run=False, desired_video_codecs=None, desired_audio_codecs=None,
                    max_resolution=None,
                    verbose=False):
    for file_info in need_transcode_generator(plex_url=plex_url, media_paths=media_paths,
                                              desired_video_codecs=desired_video_codecs,
                                              desired_audio_codecs=desired_audio_codecs, max_resolution=max_resolution):

        try:
            post_process_code = dvr_post_process(file_info.host_file_path, dry_run=dry_run, verbose=verbose,
                                                 profanity_filter=True)
        except CalledProcessError as e:
            post_process_code = e.returncode
        except FileExistsError:
            # conflict with multiple processors
            post_process_code = 255
        if post_process_code == 0 and plex_url and file_info.item_key:
            logger.info(f'HTTP PUT: {plex_url}{file_info.item_key}/analyze')
            if not dry_run:
                requests.put(f'{plex_url}{file_info.item_key}/analyze')

    return 0


def should_transcode_run():
    if common.core_count() < 4:
        return False
    return common.should_start_processing()


def transcode_apply_cli(argv):
    plex_url = common.get_plex_url()
    media_paths = None
    desired_video_codecs = None
    desired_audio_codecs = None
    max_resolution = None
    dry_run = False
    check_compute = True
    verbose = False

    if common.core_count() < 9:
        max_resolution = 480

    try:
        opts, args = getopt.getopt(list(argv),
                                   "hnu:v:a:",
                                   ["dry-run", "ignore-compute", "url=", "video=", "audio=", "maxres=", "verbose"])
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
        elif opt == '--maxres':
            max_resolution = int(arg)
        elif opt == '--verbose':
            verbose = True
        elif opt == '--ignore-compute':
            check_compute = False

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
                    desired_audio_codecs=desired_audio_codecs, max_resolution=max_resolution, verbose=verbose)
    return 0


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(transcode_apply_cli(sys.argv[1:]))
