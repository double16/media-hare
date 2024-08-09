#!/usr/bin/env python3
import getopt
import logging
import os
import sys
from collections.abc import Iterable

import common
from common import constants, edl_util


logger = logging.getLogger(__name__)


def usage():
    print(f"""{sys.argv[0]} [media_paths]

List files needing commercials cut.

Output options:

1. Absolute paths terminated with null (this can be changed) with the intent to be piped into xargs or similar tool.
2. Nagios monitoring output, which is also human readable. This also provides some estimates on time to transcode.

-t, --terminator="\\n"
    Set the output terminator, defaults to null (0).
-d, --dir=
    Directory containing media. Defaults to {common.get_media_roots()}
--nagios
    Output for Nagios monitoring. Also human readable with statistics and estimates of transcode time.
""", file=sys.stderr)


def find_need_comcut_cli(argv):
    roots = []
    terminator = '\0'
    nagios_output = False

    try:
        opts, args = getopt.getopt(argv, "t:d:",
                                   ["terminator=", "dir=", "nagios"])
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

    if not roots:
        roots = common.get_media_roots()

    if args:
        media_paths = common.get_media_paths(roots, args)
    else:
        media_paths = list(filter(lambda path: 'Movies' in path, common.get_media_paths(roots)))
    logger.debug("media_paths = %s", media_paths)

    if nagios_output:
        pending_files = list(need_comcut_generator(media_paths=media_paths, media_roots=roots))
        pending_files.sort(key=lambda e: e.size, reverse=True)
        uncut_length = sum(map(lambda e: e.uncut_length, pending_files))
        cut_length = sum(map(lambda e: e.cut_length, pending_files))

        level = "OK"
        code = 0

        print(
            f"COMCUT_PENDING {level}: files: {len(pending_files)}, uncut: {common.seconds_to_timespec(uncut_length)}, cut: {common.seconds_to_timespec(cut_length)} | COMCUT_PENDING;{len(pending_files)};{uncut_length};{cut_length}")
        for e in pending_files:
            print(
                f"{e.file_name};{e.size};{common.seconds_to_timespec(e.uncut_length)};{common.seconds_to_timespec(e.cut_length)};{e.cut_count}")
        return code
    else:
        for e in need_comcut_generator(media_paths=media_paths, media_roots=roots):
            sys.stdout.write(e.file_name)
            sys.stdout.write(terminator)
        return 0


class ComcutPendingFileInfo(object):

    def __init__(self, file_name: str, host_file_path: str, size: float, uncut_length: float, cut_length: float,
                 cut_count: int):
        self.file_name = file_name
        self.host_file_path = host_file_path
        self.size = size
        self.uncut_length = uncut_length
        self.cut_length = cut_length
        self.cut_count = cut_count


def need_comcut_generator(media_paths: list[str], media_roots: list[str]) -> Iterable[ComcutPendingFileInfo]:
    for media_path in media_paths:
        for root, dirs, files in os.walk(media_path, topdown=True):
            files_set = set(files)
            for file in common.filter_for_mkv(files):
                filepath = os.path.join(root, file)
                edl_file = file.replace('.mkv', '.bak.edl')
                if edl_file not in files_set:
                    edl_file = file.replace('.mkv', '.edl')
                if edl_file not in files_set:
                    continue
                edl_filepath = os.path.join(root, edl_file)
                if os.stat(edl_filepath).st_size < 25:
                    continue
                edl = edl_util.parse_edl_cuts(edl_filepath)
                if len(edl) == 0:
                    continue
                input_info = common.find_input_info(filepath, raise_errors=False)
                if not input_info:
                    continue
                uncut_length = float(input_info[constants.K_FORMAT][constants.K_DURATION])
                cut_length = uncut_length - sum(map(lambda e: e.length(), edl))
                file_info = ComcutPendingFileInfo(
                    file_name=common.get_media_file_relative_to_root(filepath, media_roots)[0],
                    host_file_path=filepath,
                    size=os.stat(filepath).st_size,
                    uncut_length=uncut_length,
                    cut_length=cut_length,
                    cut_count=len(edl))
                yield file_info


if __name__ == '__main__':
    common.setup_cli(level=logging.ERROR, start_gauges=False)
    sys.exit(find_need_comcut_cli(sys.argv[1:]))
