#!/usr/bin/env python3
import logging
import re
import sys

import common
from common import edl_util


def usage():
    print(f"""
Normalizes an EDL file to be more human readable. Such as converting seconds to HH:MM:SS.

Usage: {sys.argv[0]} file1 [file2 file3 ...]
""")


_EDL_NORMALIZE_RE = re.compile(r'(^|\s)(\d+\.\d+)')
_EDL_SIMPLIFY_RE = re.compile(r'(^|\s)(\d\d:\d\d:\d\d(?:\.\d+)?)')


def edl_normalize(filename, new_filename=None):
    return _edl_transform(filename, new_filename, _EDL_NORMALIZE_RE, lambda e: common.s_to_ts(float(e)))


def edl_simplify(filename, new_filename=None):
    return _edl_transform(filename, new_filename, _EDL_SIMPLIFY_RE, lambda e: str(edl_util.parse_edl_ts(e)))


def _edl_transform(filename, new_filename, regex, transform_func):
    changed = False
    new_content = ""
    with open(filename, 'rt') as edl_fd:
        for edl_line in edl_fd.readlines():
            if not edl_line.startswith("##"):
                changed_line = regex.sub(lambda m: m[1] + transform_func(m[2]), edl_line)
                if changed_line != edl_line:
                    changed = True
                new_content = new_content + changed_line
            else:
                new_content = new_content + edl_line

    if changed or new_filename:
        with open(new_filename or filename, 'wt') as edl_fd:
            edl_fd.write(new_content)

    return 0


def edl_normalize_cli(args):
    return_code = 0
    for filename in args:
        return_code = edl_normalize(filename)
        if return_code != 0:
            break
    return return_code


if __name__ == '__main__':
    common.setup_cli(level=logging.ERROR, start_gauges=False)
    sys.exit(edl_normalize_cli(sys.argv[1:]))
