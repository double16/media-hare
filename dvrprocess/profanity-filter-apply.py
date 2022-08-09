#!/usr/bin/env python3
import getopt
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from multiprocessing import Pool
from subprocess import CalledProcessError

import requests

import common
from find_need_transcode import need_transcode_generator
from profanity_filter import profanity_filter, FILTER_VERSION

logger = logging.getLogger(__name__)


def usage():
    print(f"""
Searches for content on which to apply the profanity filter.

Usage: {sys.argv[0]} [options] [media_paths|--url=.]

This program will run only if configuration profanity_filter.enable is set to true, or --force is used.
 
-n, --dry-run
-f, --force
    Override configuration profanity_filter.enable
--work-dir={common.get_work_dir()}
-b, --bytes-limit={common.get_global_config_option('background_limits', 'size_limit')}
    Limit changed data to this many bytes. Set to 0 for no limit.
-t, --time-limit={common.get_global_config_option('background_limits', 'time_limit')}
    Limit runtime. Set to 0 for no limit.
-p, --processes=2
-s, --selector={','.join(ProfanityFilterSelector.__members__.keys())}
    Limit which media files will be updated.
        unfiltered: only media that has never been filtered
        new_version: only media that has an older filter version, not common
        config_change: config or wordlist change, version change or unfiltered, basically any change at all
-u, --url=
    Find files to process from a Plex Media Server. Specify the URL such as http://127.0.0.1:32400 or '.' for {common.get_plex_url()}
""", file=sys.stderr)


class ProfanityFilterSelector(Enum):
    """ include media that was processed with a different config/wordlist """
    config_change = 0
    """ include media that hasn't been touched by the filter """
    unfiltered = 1
    """ include media that differs in version """
    new_version = 2


def __profanity_filter_selector(generator, selectors: set[ProfanityFilterSelector] = None):
    """
    Filters media items based on profanity filter properties.
    :param generator: need_transcode_generator instance
    :param selectors:
    :return: generator with same types as input, filtered by arguments
    """
    if selectors is None:
        selectors = set(ProfanityFilterSelector)

    for item in generator:
        if ProfanityFilterSelector.config_change in selectors:
            yield item
            continue

        tags = json.loads(subprocess.check_output(
            [common.find_ffprobe(), '-v', 'quiet', '-print_format', 'json', '-show_format', item.host_file_path])).get(
            'format', {}).get('tags', {})

        if common.is_truthy(tags.get(common.K_FILTER_SKIP, None)):
            continue

        if ProfanityFilterSelector.unfiltered in selectors and common.K_FILTER_HASH not in tags:
            yield item
            continue
        if ProfanityFilterSelector.new_version in selectors and common.K_FILTER_VERSION in tags and int(
                tags.get(common.K_FILTER_VERSION)) < FILTER_VERSION:
            yield item
            continue


def profanity_filter_apply(media_paths, plex_url=None, dry_run=False, workdir=None,
                           size_limit=common.get_global_config_bytes('background_limits', 'size_limit'),
                           time_limit=common.get_global_config_time_seconds('background_limits', 'time_limit'),
                           processes=1,
                           check_compute=True,
                           selectors: set[ProfanityFilterSelector] = None):
    logger.info(f"Applying profanity filter to media files in {plex_url or ','.join(media_paths)}")

    if workdir is None:
        workdir = common.get_work_dir()

    if selectors is None:
        selectors = set(ProfanityFilterSelector)

    bytes_processed = 0
    time_start = None

    generator = need_transcode_generator(plex_url=plex_url, media_paths=media_paths,
                                         # get everything
                                         desired_video_codecs=['all'],
                                         # ensures there is a subtitle stream
                                         desired_subtitle_codecs=['all'])
    generator = __profanity_filter_selector(generator, selectors)

    pool = Pool(processes=processes)
    try:
        results = []
        # load the initial workers
        for i in range(processes):
            try:
                tfi = next(generator)
                results.append([tfi, pool.apply_async(profanity_filter, (tfi.host_file_path,),
                                                      {'dry_run': dry_run, 'workdir': workdir})])
            except StopIteration:
                break

        while len(results) > 0:
            result = None
            while result is None:
                for i in range(len(results)):
                    val = results[i]
                    try:
                        val[1].wait(3)
                        results.pop(i)
                        result = val
                        break
                    except TimeoutError:
                        pass

            tfi = result[0]
            filepath = tfi.host_file_path
            try:
                return_code = result[1].get()
            except UnicodeDecodeError as e:
                # FIXME: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfe in position 1855: invalid start byte
                logger.error(e.__repr__())
                return_code = 255
            except CalledProcessError as e:
                return_code = e.returncode
            except:
                pool.terminate()
                return 255
            if return_code == 0:
                # filtered
                bytes_processed += os.stat(filepath).st_size
                if time_start is None:
                    time_start = time.time()
                # Run Plex analyze so playback works
                if tfi.item_key and plex_url:
                    logger.info(f'HTTP PUT: {plex_url}{tfi.item_key}/analyze')
                    if not dry_run:
                        try:
                            requests.put(f'{plex_url}{tfi.item_key}/analyze')
                        except requests.exceptions.ConnectTimeout:
                            pass
            elif return_code == 1:
                # marked but content unchanged
                pass
            elif return_code == 2:
                # unchanged
                pass
            elif return_code == 255:
                # error
                pass

            if 0 < size_limit < bytes_processed:
                logger.info(
                    f"Exiting normally after processing {common.bytes_to_human_str(bytes_processed)} bytes, size limit of {common.bytes_to_human_str(size_limit)} reached")
                return 0

            if time_start is not None:
                duration = time.time() - time_start
                if 0 < time_limit < duration:
                    logger.info(
                        f"Exiting normally after processing {common.s_to_ts(int(duration))}, limit of {common.s_to_ts(time_limit)} reached")
                    return 0

            if check_compute and common.should_stop_processing():
                logger.info(f"not enough compute available")
                return 0

            try:
                tfi = next(generator)
                results.append([tfi, pool.apply_async(profanity_filter, (tfi.host_file_path,),
                                                      {'dry_run': dry_run, 'workdir': workdir})])
            except StopIteration:
                pass

    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.close()
        logger.info("Waiting for pool workers to finish")
        pool.join()

    logger.info(f"Exiting normally after processing {common.bytes_to_human_str(bytes_processed)} bytes")
    return 0


def profanity_filter_apply_cli(argv):
    workdir = common.get_work_dir()
    dry_run = False
    bytes_limit = common.get_global_config_bytes('background_limits', 'size_limit')
    time_limit = common.get_global_config_time_seconds('background_limits', 'time_limit')
    check_compute = True
    plex_url = None
    force = False
    selectors = None

    processes = common.get_global_config_int('background_limits', 'processes',
                                             fallback=max(1, int(common.core_count() / 2) - 1))

    try:
        opts, args = getopt.getopt(list(argv),
                                   "fnb:t:p:u:s:",
                                   ["force", "dry-run", "work-dir=", "bytes-limit=", "time-limit=", "processes=",
                                    "url=", "selector="])
    except getopt.GetoptError:
        usage()
        return 255
    for opt, arg in opts:
        if opt == '--help':
            usage()
            return 255
        elif opt in ["-n", "--dry-run"]:
            dry_run = True
        elif opt in ["-f", "--force"]:
            force = True
        elif opt == "--work-dir":
            workdir = arg
        elif opt in ["-b", "--bytes-limit"]:
            bytes_limit = common.parse_bytes(arg)
        elif opt in ["-t", "--time-limit"]:
            time_limit = common.parse_seconds(arg)
        elif opt in ["-p", "--processes"]:
            processes = int(arg)
            check_compute = False
        elif opt in ["-u", "--url"]:
            if arg == '.' or len(arg) == 0:
                plex_url = common.get_plex_url()
            else:
                plex_url = arg
        elif opt in ["-s", "--selector"]:
            selectors = set()
            for selector_str in arg.split(','):
                selectors.add(ProfanityFilterSelector.__members__[selector_str.lower()])

    if not common.get_global_config_boolean('profanity_filter', 'enable', False) and not force:
        logger.info("profanity filter not enabled, set profanity_filter.enable to true or use --force")
        return 0

    if args:
        media_paths = args
    elif plex_url:
        media_paths = []
    else:
        media_paths = common.get_media_paths()

    if not plex_url and not media_paths:
        logger.error("No plex URL, configure in media-hare.ini, section plex, option url")
        return 255

    if selectors is None:
        selectors = set(ProfanityFilterSelector)

    if common.check_already_running():
        return 0

    if check_compute and not common.should_start_processing():
        logger.warning(f"not enough compute available")
        return 0

    return profanity_filter_apply(media_paths, plex_url=plex_url, dry_run=dry_run, workdir=workdir,
                                  size_limit=bytes_limit,
                                  time_limit=time_limit, processes=processes, check_compute=check_compute,
                                  selectors=selectors)


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(profanity_filter_apply_cli(sys.argv[1:]))
