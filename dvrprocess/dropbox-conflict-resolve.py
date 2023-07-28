#!/usr/bin/env python3
import logging
import os.path
import re
import shutil
import sys

import common
from common import constants
import profanity_filter

logger = logging.getLogger(__name__)


class Resolution(object):
    def __init__(self, preferred_path, other_path, target_path, cause: str):
        self.preferred_path = preferred_path
        self.other_path = other_path
        self.target_path = target_path
        self.cause = cause

    def resolve(self):
        self.dryrun()
        if self.preferred_path != self.target_path:
            shutil.move(self.preferred_path, self.target_path)
        elif self.preferred_path == self.target_path and self.preferred_path != self.other_path:
            os.remove(self.other_path)

    def dryrun(self):
        if self.preferred_path != self.target_path:
            logger.info("mv '%s' '%s' ; # %s", self.preferred_path, self.target_path, self.cause)
        elif self.preferred_path == self.target_path and self.preferred_path != self.other_path:
            logger.info("rm '%s' ; # %s", self.other_path, self.cause)


def resolution_generator(paths: list[str]):
    latest_filter_hash = profanity_filter.compute_filter_hash()
    latest_filter_version = profanity_filter.FILTER_VERSION
    for conflict_path, _ in common.generate_video_files(paths, fail_on_missing=False):
        if 'conflicted copy' not in conflict_path:
            continue
        logger.debug("Found %s", conflict_path)
        original_path = re.sub(r' \([^(]+ conflicted copy [0-9-]+\)', '', conflict_path)
        logger.debug("Original %s", original_path)
        if not os.path.exists(original_path):
            continue
        logger.debug("Checking %s vs. %s", conflict_path, original_path)
        conflict_info = common.find_input_info(conflict_path)
        original_info = common.find_input_info(original_path)

        conflict_has_words = has_words(conflict_info)
        original_has_words = has_words(original_info)

        if conflict_has_words and not original_has_words:
            yield Resolution(preferred_path=conflict_path, other_path=original_path, target_path=original_path,
                             cause="conflict has transcription")
        elif original_has_words and not conflict_has_words:
            yield Resolution(preferred_path=original_path, other_path=conflict_path, target_path=original_path,
                             cause="original has transcription")

        conflict_filter_version, conflict_filter_hash = get_pfilter_version_hash(conflict_info)
        original_filter_version, original_filter_hash = get_pfilter_version_hash(original_info)
        if conflict_filter_version > original_filter_version:
            yield Resolution(preferred_path=conflict_path, other_path=original_path, target_path=original_path,
                             cause="conflict has newer filter version")
        elif original_filter_version > conflict_filter_version:
            yield Resolution(preferred_path=original_path, other_path=conflict_path, target_path=original_path,
                             cause="original has newer filter version")

        if conflict_filter_hash == latest_filter_hash and original_filter_hash != latest_filter_hash:
            yield Resolution(preferred_path=conflict_path, other_path=original_path, target_path=original_path,
                             cause="conflict has latest filter hash")
        elif original_filter_hash == latest_filter_hash and conflict_filter_hash != latest_filter_hash:
            yield Resolution(preferred_path=original_path, other_path=conflict_path, target_path=original_path,
                             cause="original has latest filter hash")


def dropbox_conflict_resolve(argv: list[str]):
    if not argv:
        argv = ['.']
    for resolution in resolution_generator(argv):
        resolution.resolve()


def get_pfilter_version_hash(input_info) -> tuple[int, int]:
    filter_version = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(constants.K_FILTER_VERSION)
    filter_hash = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(constants.K_FILTER_HASH)
    return filter_version, filter_hash


def has_words(input_info) -> bool:
    words_streams = list(
        filter(lambda stream: stream.get('tags', dict()).get(constants.K_STREAM_TITLE) == constants.TITLE_WORDS,
               input_info[constants.K_STREAMS]))
    return len(words_streams) > 0


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(dropbox_conflict_resolve(sys.argv[1:]))
