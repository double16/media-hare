import logging
import os
import stat

logger = logging.getLogger(__name__)


def match_owner_and_perm(target_path: str, source_path: str) -> bool:
    result = True
    source_stat = os.stat(source_path)
    try:
        os.chown(target_path, source_stat.st_uid, source_stat.st_gid)
    except OSError:
        logger.warning(f"Changing ownership of {target_path} failed, continuing")
        result = False

    try:
        st_mode = source_stat.st_mode
        # if source is dir and has suid or guid and target is a file, mask suid/guid
        if os.path.isfile(target_path):
            st_mode &= ~(stat.S_ISUID | stat.S_ISGID | 0o111)
        os.chmod(target_path, st_mode)
    except OSError:
        logger.warning(f"Changing permission of {target_path} failed, continuing")
        result = False

    return result
