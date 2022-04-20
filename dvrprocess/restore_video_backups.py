#!/usr/bin/env python3
import logging
import os
import re
import shutil
import sys
import common

logger = logging.getLogger(__name__)


def restore_video_backups_cli(argv):
    if not argv:
        return 255

    for media_path in argv:
        for root, dirs, files in os.walk(media_path):
            for file in files:
                filepath = os.path.abspath(os.path.join(root, file))
                filename = os.path.basename(filepath)
                if filename.startswith('.~') and common.is_video_file(filepath):
                    restored_filepath = os.path.join(os.path.dirname(filepath), re.sub(r'^.~', '', filename))
                    logger.info(f"Restoring {filepath} to {restored_filepath}")
                    shutil.move(filepath, restored_filepath)


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(restore_video_backups_cli(sys.argv[1:]))
