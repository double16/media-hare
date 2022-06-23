#!/usr/bin/env python3
import getopt
import logging
import os
import sys

import dropbox
from dropbox.exceptions import ApiError

import common

logger = logging.getLogger(__name__)


def find_precut_revision(dbx, path, has_been_cut: bool):
    entries = dbx.files_list_revisions(path, limit=30).entries
    revisions = sorted(entries, key=lambda entry: entry.server_modified, reverse=True)
    latest_size = revisions[0].size
    logger.info("%s: latest_size = %s, has been cut = %s", path, common.bytes_to_human_str(latest_size),
                str(has_been_cut))
    if has_been_cut:
        target_size = latest_size * 1.07
    else:
        target_size = latest_size * 1.1
    stop_size_sm = latest_size * 0.8
    stop_size_lg = latest_size * 1.8
    sizes = list()
    for rev in revisions[1:]:
        size_str = common.bytes_to_human_str(rev.size)
        if len(sizes) == 0 or sizes[-1] != size_str:
            sizes.append(size_str)

        if rev.size > stop_size_lg:
            logger.info("%s: found pre-transcoded version, %s bytes", path, size_str)
            return None
        if rev.size > target_size:
            logger.info("%s: found pre-cut version, %s bytes", path, size_str)
            return rev
        if rev.size < stop_size_sm:
            logger.info("%s: found cut version, %s bytes, looks like current version is pre-cut", path, size_str)
            return None

    logger.info("%s: no pre-cut version found from sizes %s", path, sizes)
    return None


def generate_files(media_paths: list[str]):
    for media_path in media_paths:
        if os.path.isfile(media_path):
            if common.is_video_file(media_path):
                yield os.path.abspath(media_path)
        for root, dirs, files in os.walk(media_path):
            files.sort()
            for file in files:
                filepath = os.path.abspath(os.path.join(root, file))
                filename = os.path.basename(filepath)
                if not filename.startswith('.') and common.is_video_file(filepath):
                    yield filepath


def comcut_restore_cli(argv):
    dry_run = False

    try:
        opts, args = getopt.getopt(list(argv),
                                   "hnv",
                                   ["help", "verbose", "dry-run"])
    except getopt.GetoptError:
        usage()
        return 255
    for opt, arg in opts:
        if opt in ['-h', '--help']:
            usage()
            return 255
        elif opt == "--verbose":
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt in ["-n", "--dry-run"]:
            dry_run = True

    if len(args) == 0:
        sys.exit("ERROR: Expected list of files or directories")

    token = os.getenv('DROPBOX_ACCESS_TOKEN')
    if not token:
        sys.exit("ERROR: Access token required in DROPBOX_ACCESS_TOKEN")

    with dropbox.Dropbox(token) as dbx:
        media_base = common.find_media_base()
        for video_filepath in generate_files(args):
            edl_filepath = common.edl_for_video(video_filepath)
            has_been_cut = False
            if edl_filepath and os.path.isfile(edl_filepath):
                with open(edl_filepath, 'r') as f:
                    has_been_cut = '## cut complete' in f.read()
            dropbox_path = video_filepath.replace(media_base, '/Media')
            logger.info(dropbox_path)
            uncut_rev = find_precut_revision(dbx, dropbox_path, has_been_cut)
            if uncut_rev is not None:
                logger.info("Restoring %s, size %s", dropbox_path, common.bytes_to_human_str(uncut_rev.size))
                if not dry_run:
                    try:
                        dbx.files_restore(dropbox_path, uncut_rev.rev)
                        try:
                            dbx.files_delete_v2(common.replace_extension(dropbox_path, 'edl'))
                        except ApiError:
                            pass
                        try:
                            dbx.files_delete_v2(common.replace_extension(dropbox_path, 'bak.edl'))
                        except ApiError:
                            pass
                    except ApiError as e:
                        logger.error(f"Restoring {dropbox_path} revision {uncut_rev.rev}, {repr(e)}")

    return 0


def usage():
    print(f"""
Restore previous uncut version of files. Looks at video files with at least 15% cut in prior version.

Usage: {sys.argv[0]} file | dir

--verbose
-n, --dry-run
""", file=sys.stderr)


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(comcut_restore_cli(sys.argv[1:]))
