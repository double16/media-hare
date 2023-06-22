#!/usr/bin/env python3
import atexit
import getopt
import logging
import os
import re
import sys

import common
from common import tools, edl_util

#
# Important info on seeking: https://trac.ffmpeg.org/wiki/Seeking
#

logger = logging.getLogger(__name__)


def usage():
    print(f"""
Extracts scenes from a video using an EDL file. Cuts specified using type "2" are scenes. Text after the "2" is
considered a title and will be used to name the file. If S00E00 type information is found, the title string will
be used for the entire file.

Usage: {sys.argv[0]} infile [outfile pattern]
--dry-run
--verbose
""", file=sys.stderr)


@common.finisher
def scene_extract(infile, outfile_pattern, verbose=False, dry_run=False):
    infile_base = '.'.join(os.path.basename(infile).split('.')[0:-1])
    edlfile = f"{os.path.dirname(infile) or '.'}/{infile_base}.edl"

    outextension = outfile_pattern.split('.')[-1]
    outfile_dir = os.path.dirname(os.path.abspath(outfile_pattern)) or "."
    outfile_base = '.'.join(os.path.basename(outfile_pattern).split('.')[0:-1])

    if not os.access(edlfile, os.R_OK):
        logger.fatal(f"file not found {edlfile}")
        return 255

    keyframes = common.load_keyframes_by_seconds(infile)

    edl_events = edl_util.parse_edl(edlfile)

    part_num = 1

    for edl_event in filter(lambda e: e.event_type == edl_util.EdlType.SCENE, edl_events):
        start = edl_event.start
        end = edl_event.end
        start = common.find_desired_keyframe(keyframes, start, common.KeyframeSearchPreference.CLOSEST)
        end = common.find_desired_keyframe(keyframes, end, common.KeyframeSearchPreference.CLOSEST)
        logger.debug(f"start {common.s_to_ts(edl_event.start)} moved to {common.s_to_ts(start)}")
        logger.debug(f"end {common.s_to_ts(edl_event.end)} moved to {common.s_to_ts(end)}")
        title = edl_event.title
        if not title:
            scene_file = os.path.join(outfile_dir, f"{outfile_base} - Part {part_num}.{outextension}")
            part_num += 1
        else:
            title_sanitized_for_filesystem = re.sub(r"['\"?:!]", " ", title)
            if re.search(r"S\d+E\d+", title):
                scene_file = os.path.join(outfile_dir, f"{title_sanitized_for_filesystem}.{outextension}")
            else:
                scene_file = os.path.join(outfile_dir,
                                         f"{outfile_base} - {title_sanitized_for_filesystem}.{outextension}")

        parts_file = os.path.join(outfile_dir, f"{scene_file}.parts.txt")
        with open(parts_file, "w") as partsfd:
            partsfd.write("ffconcat version 1.0\n")
            partsfd.write("file %s\n" % re.sub('([^A-Za-z0-9/])', r'\\\1', infile))
            partsfd.write(f"inpoint {start}\n")
            partsfd.write(f"outpoint {end}\n")

        ffmpeg_command = ["-hide_banner", "-loglevel",
                          "info" if verbose else "error",
                          "-nostdin",
                          "-f", "concat", "-safe", "0", "-i", parts_file,
                          "-codec", "copy", "-map", "0",
                          '-max_muxing_queue_size', '1024',
                          '-async', '1',
                          '-max_interleave_delta', '0',
                          "-avoid_negative_ts", "1",
                          "-y"]

        if title:
            ffmpeg_command.extend(["-metadata", f'title={title}'])

        ffmpeg_command.append(scene_file)

        try:
            if dry_run:
                logger.info(tools.ffmpeg.array_as_command(ffmpeg_command))
            else:
                tools.ffmpeg.run(ffmpeg_command, check=True)
        finally:
            os.remove(parts_file)

    return 0


def scene_extract_cli(argv):
    verbose = False
    dry_run = False

    try:
        opts, args = getopt.getopt(argv, "n",
                                   ["dry-run", "verbose"])
    except getopt.GetoptError:
        usage()
        return 255
    for opt, arg in opts:
        if opt == '--help':
            usage()
            return 255
        elif opt == "--verbose":
            verbose = True
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt in ["-n", "--dry-run"]:
            dry_run = True

    if not args:
        usage()
        sys.exit(255)

    atexit.register(common.finish)

    if len(args) == 2:
        # check special case of input file and output file
        if os.path.isfile(args[0]) and (os.path.isfile(args[1]) or not os.path.exists(args[1])):
            return scene_extract(args[0], args[1], verbose=verbose, dry_run=dry_run)

    return_code = 0
    for arg in args:
        if os.path.isfile(arg):
            this_file_return_code = scene_extract(arg, arg, verbose=verbose, dry_run=dry_run)
            if this_file_return_code != 0 and return_code == 0:
                return_code = this_file_return_code
        for root, dirs, files in os.walk(arg):
            for file in files:
                filepath = os.path.join(root, file)
                filename = os.path.basename(filepath)
                if not filename.endswith(".mkv") or filename.startswith('.'):
                    continue
                if os.path.isfile(filepath.replace('.mkv', '.edl')):
                    this_file_return_code = scene_extract(filepath, verbose=verbose, dry_run=dry_run)
                    if this_file_return_code != 0 and return_code == 0:
                        return_code = this_file_return_code

    return return_code


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(scene_extract_cli(sys.argv[1:]))
