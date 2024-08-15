#!/usr/bin/env python3
import getopt
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterable

import common
from common import tools, config

logger = logging.getLogger(__name__)

ERROR_THRESHOLD = 300


def usage():
    print(f"""{sys.argv[0]} [media_paths]

List media files with errors that exceed a threshold.

Output options:

1. Absolute paths terminated with null (this can be changed) with the intent to be piped into xargs or similar tool.
2. Nagios monitoring output, which is also human readable. This also provides some estimates on time to transcode.

-t, --terminator="\\n"
    Set the output terminator, defaults to null (0).
-d, --dir=
    Directory containing media. Defaults to {common.get_media_roots()}
--nagios
    Output for Nagios monitoring. Also human readable with statistics and estimates of transcode time.
--cache-only
    Only report cached results, do not look for new media errors.
--time-limit={config.get_global_config_option('background_limits', 'time_limit')}
    Limit runtime. Set to 0 for no limit.
--ignore-compute
    Ignore current compute availability.
""", file=sys.stderr)


def find_media_errors_cli(argv):
    roots = []
    terminator = '\0'
    nagios_output = False
    time_limit = config.get_global_config_time_seconds('background_limits', 'time_limit')
    check_compute = True
    cache_only = False

    try:
        opts, args = getopt.getopt(argv, "t:d:",
                                   ["terminator=", "dir=", "nagios", "time-limit=", "ignore-compute", "cache-only"])
    except getopt.GetoptError:
        usage()
        return 2
    for opt, arg in opts:
        if opt == '-h':
            usage()
            return 2
        elif opt in ("-d", "--dir"):
            roots.append(arg)
        elif opt == '--nagios':
            nagios_output = True
        elif opt in ("-t", "--terminator"):
            if arg == '\\n':
                terminator = '\n'
            elif arg == '\\0':
                terminator = '\0'
            else:
                terminator = arg
        elif opt in ['--time-limit']:
            time_limit = config.parse_seconds(arg)
        elif opt == '--ignore-compute':
            check_compute = False
        elif opt == '--cache-only':
            cache_only = True

    if not roots:
        roots = common.get_media_roots()

    if args:
        media_paths = common.get_media_paths(roots, args)
    else:
        media_paths = common.get_media_paths(roots)
    logger.debug("media_paths = %s", media_paths)

    if common.check_already_running(quiet=True):
        cache_only = True

    generator = media_errors_generator(media_paths=media_paths, media_roots=roots,
                                       time_limit=time_limit, check_compute=check_compute, cache_only=cache_only)

    if nagios_output:
        corrupt_files = list(generator)
        corrupt_files.sort(key=lambda e: e.error_count, reverse=True)

        if len(corrupt_files) > 25:
            level = "CRITICAL"
            code = 2
        elif len(corrupt_files) > 0:
            level = "WARNING"
            code = 1
        else:
            level = "OK"
            code = 0

        print(f"MEDIA_ERRORS {level}: files: {len(corrupt_files)} | MEDIA_ERRORS;{len(corrupt_files)}")
        for e in corrupt_files:
            print(f"{e.file_name};{e.error_count}")
        return code
    else:
        for e in generator:
            sys.stdout.write(e.file_name)
            sys.stdout.write(terminator)
        return 0


class MediaErrorFileInfo(object):

    def __init__(self, file_name: str, host_file_path: str, size: float, error_count: int):
        self.file_name = file_name
        self.host_file_path = host_file_path
        self.size = size
        self.error_count = error_count


def media_errors_generator(media_paths: list[str], media_roots: list[str],
                           time_limit=config.get_global_config_time_seconds('background_limits', 'time_limit'),
                           check_compute=True, cache_only=False) -> Iterable[MediaErrorFileInfo]:
    time_start = time.time()

    for media_path in media_paths:
        for root, dirs, files in os.walk(media_path, topdown=True):
            for file in common.filter_for_mkv(files):
                filepath = os.path.join(root, file)
                cached_error_count = config.get_file_config_option(filepath, 'error', 'count')
                if cached_error_count:
                    error_count = int(cached_error_count)
                elif cache_only:
                    continue
                else:
                    duration = time.time() - time_start
                    if 0 < time_limit < duration:
                        logger.debug(
                            f"Time limit expired after processing {common.s_to_ts(int(duration))}, limit of {common.s_to_ts(time_limit)} reached, only using cached data")
                        cache_only = True
                        continue
                    if check_compute and common.should_stop_processing():
                        # when compute limit is reached, use cached data
                        logger.debug("not enough compute available, only using cached data")
                        cache_only = True
                        continue

                    error_count = len(tools.ffmpeg.check_output(
                        ['-y', '-v', 'error', '-i', filepath, '-c:v', 'vnull', '-c:a', 'anull', '-f', 'null',
                         '/dev/null'],
                        stderr=subprocess.STDOUT, text=True).splitlines())
                    config.set_file_config_option(filepath, 'error', 'count', str(error_count))
                if error_count <= ERROR_THRESHOLD:
                    continue
                file_info = MediaErrorFileInfo(
                    file_name=common.get_media_file_relative_to_root(filepath, media_roots)[0],
                    host_file_path=filepath,
                    size=os.stat(filepath).st_size,
                    error_count=error_count)
                yield file_info


if __name__ == '__main__':
    os.nice(15)
    common.setup_cli(level=logging.ERROR, start_gauges=False)
    sys.exit(find_media_errors_cli(sys.argv[1:]))
