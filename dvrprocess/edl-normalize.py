#!/usr/bin/env python3
import re
import sys

import common


def usage():
    print(f"""
Normalizes an EDL file to be more human readable. Such as converting seconds to HH:MM:SS.

Usage: {sys.argv[0]} file1 [file2 file3 ...]
""")


def edl_normalize(filename):
    changed = False
    new_content = ""
    with open(filename, 'r') as edl_fd:
        for edl_line in edl_fd.readlines():
            changed_line = re.sub(r'(^|\s)(\d+\.\d+)', lambda e: e[1] + common.s_to_ts(float(e[2])), edl_line)
            if changed_line != edl_line:
                changed = True
            new_content = new_content + changed_line

    if changed:
        with open(filename, 'w') as edl_fd:
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
    common.setup_cli()
    sys.exit(edl_normalize_cli(sys.argv[1:]))
