#!/usr/bin/env python3
import logging
import os
import sys
import atexit
import getopt
import subprocess
import re

import common

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
    ffmpeg = common.find_ffmpeg()

    infile_base = '.'.join(os.path.basename(infile).split('.')[0:-1])
    edlfile = f"{os.path.dirname(infile) or '.'}/{infile_base}.edl"

    outextension = outfile_pattern.split('.')[-1]
    outfile_dir = os.path.dirname(os.path.abspath(outfile_pattern)) or "."
    outfile_base = '.'.join(os.path.basename(outfile_pattern).split('.')[0:-1])

    if not os.access(edlfile, os.R_OK):
        logger.fatal(f"file not found {edlfile}")
        return 255

    keyframes = common.load_keyframes_by_seconds(infile)

    edl_events = common.parse_edl(edlfile)

    part_num = 1

    for edl_event in filter(lambda e: e.event_type == common.EdlType.SCENE, edl_events):
        start = edl_event.start
        end = edl_event.end
        start = common.find_desired_keyframe(keyframes, start)
        end = common.find_desired_keyframe(keyframes, end)
        logger.debug(f"start {common.s_to_ts(edl_event.start)} moved to {common.s_to_ts(start)}")
        logger.debug(f"end {common.s_to_ts(edl_event.end)} moved to {common.s_to_ts(end)}")
        title = edl_event.title
        if not title:
            scenefile = os.path.join(outfile_dir, f"{outfile_base} - Part {part_num}.{outextension}")
            part_num += 1
        else:
            title_sanitized_for_filesystem = re.sub(r"['\"?:!]", " ", title)
            if re.search(r"S\d+E\d+", title):
                scenefile = os.path.join(outfile_dir, f"{title_sanitized_for_filesystem}.{outextension}")
            else:
                scenefile = os.path.join(outfile_dir,
                                         f"{outfile_base} - {title_sanitized_for_filesystem}.{outextension}")

        ffmpeg_command = [ffmpeg, "-hide_banner", "-loglevel",
                          "info" if verbose else "error",
                          "-nostdin",
                          "-ss", str(start+0.3),
                          "-i", infile,
                          "-t", str(end - start),
                          "-codec", "copy", "-map", "0", "-avoid_negative_ts", "1", "-y"]

        if title:
            ffmpeg_command.extend(["-metadata", f'title={title}'])

        ffmpeg_command.append(scenefile)

        if dry_run:
            logger.info(common.array_as_command(ffmpeg_command))
        else:
            subprocess.run(ffmpeg_command, check=True, capture_output=not verbose)

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
