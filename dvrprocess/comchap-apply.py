#!/usr/bin/env python3
import getopt
import logging
import os
import random
import sys
import time
from multiprocessing import Pool
from subprocess import CalledProcessError

import common
from comchap import comchap, find_comskip_ini, compute_comskip_ini_hash

logger = logging.getLogger(__name__)


def usage():
    print(f"""
Searches for content on which to apply the commercial skip chapters. Never changes videos with chapters not put
there by this program.

Usage: {sys.argv[0]} [options] media_paths ...

--comskip-ini=comskip.ini
--work-dir=/tmp
-b, --bytes-limit={common.get_global_config_bytes('background_limits', 'size_limit')}
    Limit changed data to this many bytes. Set to 0 for no limit.
-t, --time-limit={common.get_global_config_time_seconds('background_limits', 'time_limit')}
    Limit runtime. Set to 0 for no limit.
--processes=2
--dry-run
--keep-log
--force
    Force application of chapters regardless if the file already has chapters.
""", file=sys.stderr)


def comchap_apply(media_paths, dry_run=False, comskip_ini=None, workdir=None, force=False,
                  size_limit=common.get_global_config_bytes('background_limits', 'size_limit'),
                  time_limit=common.get_global_config_time_seconds('background_limits', 'time_limit'),
                  processes=1,
                  check_compute=True,
                  delete_log=True):
    logger.info(f"Applying comchap to media files in {','.join(media_paths)}")

    if workdir is None:
        workdir = common.get_work_dir()

    if not comskip_ini:
        try:
            comskip_ini = find_comskip_ini()
        except OSError as e:
            logger.fatal(f"finding {comskip_ini}", exc_info=e)
            return 255

    # Limit the number of bytes we change to prevent overloading the upload bandwidth
    bytes_processed = 0
    time_start = None

    with Pool(processes=processes) as pool:
        random.shuffle(media_paths)
        for media_path in media_paths:
            for root, dirs, files in os.walk(media_path, topdown=True):
                random.shuffle(dirs)
                random.shuffle(files)
                is_show = "Shows" in root or (len(files) > 0 and "Shows" in files[0])

                filepaths = []
                for file in common.filter_for_mkv(files):
                    filepath = os.path.join(root, file)
                    input_info = common.find_input_info(filepath)
                    if not input_info:
                        continue

                    # check for existing chapters
                    remove_edl = force
                    if not force and len(input_info.get('chapters', [])) > 0:
                        current_comskip_hash = input_info.get(common.K_FORMAT, {}).get(common.K_TAGS, {}).get(
                            common.K_COMSKIP_HASH)
                        if not current_comskip_hash:
                            # we did not add these chapters
                            logger.info(f"{filepath}: Chapters from another source present")
                            continue
                        if current_comskip_hash == compute_comskip_ini_hash(comskip_ini, video_path=filepath,
                                                                            workdir=workdir):
                            # already done
                            logger.info(f"{filepath}: Already processed")
                            continue
                        else:
                            remove_edl = True

                    if remove_edl:
                        try:
                            os.remove(filepath.replace('.mkv', '.edl'))
                        except FileNotFoundError:
                            pass

                    filepaths.append(filepath)

                worker_limit = max(1, len(files)) if is_show else processes
                for filepath_group in common.split_every(worker_limit, filepaths):
                    results = []
                    for filepath in filepath_group:
                        logger.info(f"{filepath}: Looking for commercials")
                        if not dry_run:
                            results.append([filepath, pool.apply_async(comchap, (filepath, filepath),
                                                                       {'verbose': dry_run, 'delete_edl': False,
                                                                        'modify_video': not dry_run,
                                                                        'delete_log': delete_log,
                                                                        'delete_txt': delete_log,
                                                                        'workdir': workdir,
                                                                        'comskipini': comskip_ini})])

                    for result in results:
                        result[1].wait()

                    for result in results:
                        filepath = result[0]
                        try:
                            return_code = result[1].get()
                        except CalledProcessError as e:
                            return_code = e.returncode
                        if return_code == 0:
                            # processed
                            bytes_processed += os.stat(filepath).st_size
                            if time_start is None:
                                time_start = time.time()
                        elif return_code == 255:
                            # error
                            pass

                        if not is_show:
                            # Only episodic collections should complete an entire season
                            if 0 < size_limit < bytes_processed:
                                break
                            if time_start is not None:
                                duration = time.time() - time_start
                                if 0 < time_limit < duration:
                                    break

                if 0 < size_limit < bytes_processed:
                    logger.info(
                        f"Exiting normally after processing {bytes_processed} bytes, size limit of {size_limit} reached")
                    return 0

                if time_start is not None:
                    duration = time.time() - time_start
                    if 0 < time_limit < duration:
                        logger.info(
                            f"Exiting normally after processing {int(duration)}s, limit of {time_limit}s reached")
                        return 0

                if check_compute and common.should_stop_processing():
                    logger.info(f"INFO: not enough compute available")
                    return 0

    logger.info(f"Exiting normally after processing {bytes_processed} bytes")
    return 0


def comchap_apply_cli(argv):
    workdir = common.get_work_dir()
    comskipini = None
    dry_run = False
    force = False
    bytes_limit = common.get_global_config_bytes('background_limits', 'size_limit')
    time_limit = common.get_global_config_time_seconds('background_limits', 'time_limit')
    check_compute = True
    delete_log = True

    processes = common.get_global_config_int('background_limits', 'processes',
                                             fallback=max(1, int(common.core_count() / 3) - 1))

    try:
        opts, args = getopt.getopt(list(argv), "nfb:t:p:",
                                   ["dry-run", "force", "comskip-ini=", "work-dir=", "bytes-limit=", "time-limit=",
                                    "processes=", "keep-log"])
    except getopt.GetoptError:
        usage()
        return 255
    for opt, arg in opts:
        if opt == '--help':
            usage()
            return 255
        elif opt in ["-n", "--dry-run"]:
            dry_run = True
        elif opt in ["-f", "--force"]:
            force = True
        elif opt == "--comskip-ini":
            comskipini = arg
            if not os.access(comskipini, os.R_OK):
                logger.fatal(f"Cannot find comskip.ini at {comskipini}")
                return 255
        elif opt == "--work-dir":
            workdir = arg
        elif opt in ["-b", "--bytes-limit"]:
            bytes_limit = common.parse_bytes(arg)
        elif opt in ["-t", "--time-limit"]:
            time_limit = common.parse_seconds(arg)
        elif opt in ["-p", "--processes"]:
            processes = int(arg)
            check_compute = False
        elif opt == "--keep-log":
            delete_log = False

    if args:
        media_paths = args
    else:
        media_paths = common.get_media_paths()

    if common.check_already_running():
        return 0

    if check_compute and not common.should_start_processing():
        logger.warning(f"not enough compute available")
        return 255

    return comchap_apply(media_paths, dry_run=dry_run, force=force, comskip_ini=comskipini, workdir=workdir,
                         size_limit=bytes_limit, time_limit=time_limit, processes=processes,
                         check_compute=check_compute, delete_log=delete_log)


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(comchap_apply_cli(sys.argv[1:]))
