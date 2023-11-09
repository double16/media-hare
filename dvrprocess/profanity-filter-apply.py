#!/usr/bin/env python3
import getopt
import logging
import os
import sys
import time
from enum import Enum
from math import ceil
from multiprocessing import Pool, TimeoutError
from subprocess import CalledProcessError

import requests

import common
from common import constants, config, progress
from find_need_transcode import need_transcode_generator
from profanity_filter import profanity_filter, is_filter_version_outdated, compute_filter_hash

logger = logging.getLogger(__name__)


def usage():
    print(f"""
Searches for content on which to apply the profanity filter.

Usage: {sys.argv[0]} [options] [media_paths|--url=.]

This program will run only if configuration profanity_filter.enable is set to true, or --force is used.

-n, --dry-run
-f, --force
    Override configuration profanity_filter.enable
--work-dir={config.get_work_dir()}
-b, --bytes-limit={config.get_global_config_option('background_limits', 'size_limit')}
    Limit changed data to this many bytes. Set to 0 for no limit.
-t, --time-limit={config.get_global_config_option('background_limits', 'time_limit')}
    Limit runtime. Set to 0 for no limit.
-p, --processes=2
--ignore-compute
    Ignore current compute availability.
-s, --selector={','.join(ProfanityFilterSelector.__members__.keys())}
    Limit which media files will be updated.
        unfiltered: only media that has never been filtered
        new_version: only media that has an older filter version, not common
        config_change: config or wordlist change, version change or unfiltered, basically any change at all
-u, --url=
    Find files to process from a Plex Media Server. Specify the URL such as http://127.0.0.1:32400 or '.' for {common.get_plex_url()}
--verbose
    Verbose information about the process
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

    filter_hash = compute_filter_hash()

    if selectors is None:
        selectors = set(ProfanityFilterSelector)

    queue_max_size = 500
    queue_new_version = []
    queue_config_change = []

    item_progress = progress.progress("files", 0, 0)
    item_progress.renderer = lambda pos: f"{pos}/{item_progress.range()[1]}"

    for item in generator:
        if common.is_truthy(item.tags.get(constants.K_FILTER_SKIP, None)):
            continue

        if constants.K_FILTER_HASH not in item.tags:
            if ProfanityFilterSelector.unfiltered in selectors:
                item_progress.progress_inc(value=1, end_inc=1)
                yield item

        elif is_filter_version_outdated(item.tags):
            if ProfanityFilterSelector.new_version in selectors:
                if len(queue_new_version) < queue_max_size:
                    queue_new_version.append(item)
                    item_progress.end_inc()

        elif filter_hash != item.tags.get(constants.K_FILTER_HASH, ""):
            if ProfanityFilterSelector.config_change in selectors:
                if len(queue_config_change) < queue_max_size:
                    queue_config_change.append(item)
                    item_progress.end_inc()

    while queue_new_version:
        item_progress.progress_inc()
        yield queue_new_version.pop(0)
    while queue_config_change:
        item_progress.progress_inc()
        yield queue_config_change.pop(0)

    item_progress.stop()


def profanity_filter_apply(media_paths, plex_url=None, dry_run=False, workdir=None,
                           size_limit=config.get_global_config_bytes('background_limits', 'size_limit'),
                           time_limit=config.get_global_config_time_seconds('background_limits', 'time_limit'),
                           processes=1,
                           check_compute=True,
                           selectors: set[ProfanityFilterSelector] = None):
    logger.info(f"Applying profanity filter to media files in {plex_url or ','.join(media_paths)}")

    if workdir is None:
        workdir = config.get_work_dir()

    if selectors is None:
        selectors = set(ProfanityFilterSelector)

    bytes_processed = 0
    bytes_progress = progress.progress("byte limit", 0, size_limit)
    bytes_progress.renderer = config.bytes_to_human_str
    time_start = time.time()
    time_progress = progress.progress("time limit", 0, time_limit)
    time_progress.renderer = common.s_to_ts

    generator = need_transcode_generator(plex_url=plex_url, media_paths=media_paths,
                                         # get everything
                                         desired_video_codecs=['all'],
                                         # ensures there is a subtitle stream
                                         desired_subtitle_codecs=['all'])
    generator = __profanity_filter_selector(generator, selectors)

    if dry_run:
        for tfi in generator:
            logger.info("Processing %s", tfi.host_file_path)
        return 0

    pool = Pool(processes=processes, maxtasksperchild=10)
    try:
        results = []
        # load the initial workers
        for i in range(processes):
            try:
                tfi = next(generator)
                results.append([tfi, pool.apply_async(common.pool_apply_wrapper(profanity_filter),
                                                      (tfi.host_file_path,),
                                                      {'dry_run': dry_run, 'workdir': workdir})])
            except StopIteration:
                break

        while len(results) > 0:
            for i in range(len(results)):
                if i >= len(results):  # results may have changed
                    break
                result = results[i]
                tfi = result[0]
                filepath = tfi.host_file_path
                try:
                    return_code = result[1].get(3)
                except TimeoutError:
                    continue
                except UnicodeDecodeError as e:
                    # FIXME: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfe in position 1855: invalid start byte
                    logger.error(e.__repr__(), exc_info=e)
                    return_code = 255
                except CalledProcessError as e:
                    return_code = e.returncode
                except Exception as e:
                    logger.error(e.__repr__(), exc_info=e)
                    pool.terminate()
                    return 255

                results.pop(i)

                if return_code == 0:
                    # filtered
                    bytes_processed += os.stat(filepath).st_size
                    bytes_progress.progress(bytes_processed)
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
                        f"Exiting normally after processing {config.bytes_to_human_str(bytes_processed)} bytes, size limit of {config.bytes_to_human_str(size_limit)} reached")
                    return 0

                duration = time.time() - time_start
                time_progress.progress(ceil(duration))
                if 0 < time_limit < duration:
                    logger.info(
                        f"Exiting normally after processing {common.s_to_ts(int(duration))}, limit of {common.s_to_ts(time_limit)} reached")
                    return 0

                if check_compute and common.should_stop_processing():
                    logger.info(f"not enough compute available")
                    return 0

                try:
                    tfi = next(generator)
                    results.append(
                        [tfi, pool.apply_async(common.pool_apply_wrapper(profanity_filter), (tfi.host_file_path,),
                                               {'dry_run': dry_run, 'workdir': workdir})])
                except StopIteration:
                    pass

    except KeyboardInterrupt:
        try:
            pool.close()
            logger.info("Waiting for pool workers to finish, interrupt again to terminate")
            pool.join()
        except KeyboardInterrupt:
            logger.info("Terminating due to user interrupt")
            pool.terminate()
            return 130
    finally:
        pool.close()
        logger.info("Waiting for pool workers to finish")
        pool.join()
        bytes_progress.stop()
        time_progress.stop()

    logger.info(f"Exiting normally after processing {config.bytes_to_human_str(bytes_processed)} bytes")
    return 0


def profanity_filter_apply_cli(argv) -> int:
    workdir = config.get_work_dir()
    no_curses = False
    dry_run = False
    bytes_limit = config.get_global_config_bytes('background_limits', 'size_limit')
    time_limit = config.get_global_config_time_seconds('background_limits', 'time_limit')
    check_compute = True
    plex_url = None
    force = False
    selectors = None

    processes = config.get_global_config_int('background_limits', 'processes',
                                             fallback=max(1, int(common.core_count() / 2) - 1))

    try:
        opts, args = getopt.getopt(list(argv),
                                   "fnb:t:p:u:s:",
                                   ["force", "dry-run", "work-dir=", "bytes-limit=", "time-limit=", "processes=",
                                    "url=", "selector=", "verbose", "no-curses", "ignore-compute"])
    except getopt.GetoptError:
        usage()
        return 255
    for opt, arg in opts:
        if opt == '--help':
            usage()
            return 255
        elif opt in ["-n", "--dry-run"]:
            dry_run = True
            no_curses = True
        elif opt in ["-f", "--force"]:
            force = True
        elif opt == "--work-dir":
            workdir = arg
        elif opt in ["-b", "--bytes-limit"]:
            bytes_limit = config.parse_bytes(arg)
        elif opt in ["-t", "--time-limit"]:
            time_limit = config.parse_seconds(arg)
        elif opt in ["-p", "--processes"]:
            processes = int(arg)
            check_compute = False
        elif opt == '--ignore-compute':
            check_compute = False
        elif opt in ["-u", "--url"]:
            if arg == '.' or len(arg) == 0:
                plex_url = common.get_plex_url()
            else:
                plex_url = arg
        elif opt == "--verbose":
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt == "--no-curses":
            no_curses = True
        elif opt in ["-s", "--selector"]:
            selectors = set()
            for selector_str in arg.split(','):
                selectors.add(ProfanityFilterSelector.__members__[selector_str.lower()])

    if not config.get_global_config_boolean('profanity_filter', 'enable', False) and not force:
        print("profanity filter not enabled, set profanity_filter.enable to true or use --force", file=sys.stderr)
        return 0

    if args:
        media_paths = args
    elif plex_url:
        media_paths = []
    else:
        media_paths = common.get_media_paths()

    if not plex_url and not media_paths:
        print("No plex URL, configure in media-hare.ini, section plex, option url", file=sys.stderr)
        return 255

    if selectors is None:
        selectors = set(ProfanityFilterSelector)

    if common.check_already_running():
        return 0

    if check_compute and not common.should_start_processing():
        print("not enough compute available", file=sys.stderr)
        return 0

    return common.cli_wrapper(
        profanity_filter_apply, media_paths, plex_url=plex_url, dry_run=dry_run, workdir=workdir,
        size_limit=bytes_limit,
        time_limit=time_limit, processes=processes, check_compute=check_compute,
        selectors=selectors, no_curses=no_curses)


if __name__ == '__main__':
    sys.exit(profanity_filter_apply_cli(sys.argv[1:]))
