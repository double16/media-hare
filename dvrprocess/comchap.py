#!/usr/bin/env python3
import atexit
import configparser
import getopt
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile

import common
from common import hwaccel

logger = logging.getLogger(__name__)

INI_GROUP_MAIN_SETTINGS = 'Main Settings'
INI_GROUP_MAIN_SCORING = 'Main Scoring'
INI_GROUP_DETAILED_SETTINGS = 'Detailed Settings'
INI_GROUP_OUTPUT_CONTROL = 'Output Control'
INI_GROUP_GLOBAL_REMOVES = 'Global Removes'
INI_GROUP_LOGO_FINDING = 'Logo Finding'
INI_GROUP_LOGO_INTERPRETATION = 'Logo Interpretation'
INI_GROUP_VERSIONS = 'Versions'
INI_ITEM_VERSIONS_VIDEO_STATS = 'video_stats'
INI_ITEM_VERSIONS_GAD_TUNING = 'gad_tuning'


def usage():
    print(f"""
Add chapter and commercial markers to a video file using an EDL file.
If no EDL file is found, comskip will be used to generate one.
MKV files are optimized in that ffmpeg is not necessary to remux, mkvpropedit will be used if found.

Usage: {sys.argv[0]} infile [outfile]

--keep-edl
    Keep the edl in addition to adding chapters to the video
--only-edl
    Only generate an edl file
--keep-meta
    Keep the meta data file used to add chapters to the video
--keep-log
    Keep the comskip log, if comskip is run
--keep-ini
    Keep the fully assembled ini file
-k, --keep-all
--verbose
--comskip-ini=
--work-dir={common.get_work_dir()}
--mark-skip
    Mark file(s) to skip commercial scanning. Existing commercial markers, EDL and comskip.ini files will be removed.
--force
    Force re-running comskip even if an edl file is present
--debug
    Open the comskip debug window
""", file=sys.stderr)


def find_comskip_ini():
    return common.find_config('comskip.ini')


def get_expected_adjusted_duration(video_info):
    """
    Get the expected duration of a show, i.e. without commercials. The video is assumed to be from DVR, no
    check is done to verify.
    :param video_info:
    :return:
    """
    return common.round_episode_duration(video_info) - get_added_recording_seconds(video_info)


def get_added_recording_seconds(video_info):
    """
    Get the number of seconds of added content to a show, i.e. commercials. The video is assumed to be from DVR, no
    check is done to verify.
    :param video_info:
    :return:
    """
    # This is the full config, including previously tuned parameters
    comskip_config = configparser.ConfigParser()
    comskip_config_file = build_comskip_ini(find_comskip_ini(), input_info=video_info, workdir=tempfile.gettempdir())
    comskip_config.read(comskip_config_file)
    os.remove(comskip_config_file)

    added_recording_minutes = float(comskip_config.get(INI_GROUP_GLOBAL_REMOVES, 'added_recording', fallback='0.0'))
    if added_recording_minutes > 0.0:
        # There's always -1 to account for DVR shorting the recording by 1-60 seconds
        added_recording_minutes += 1.0
    return added_recording_minutes * 60.0


def compute_comskip_ini_hash(comskip_ini, leaf_comskip_ini=None, video_path=None, input_info=None,
                             workdir=tempfile.gettempdir()):
    final_comskip_ini = build_comskip_ini(comskip_ini, leaf_comskip_ini=leaf_comskip_ini, video_path=video_path,
                                          input_info=input_info, workdir=workdir)
    try:
        hash_input = ""
        with open(final_comskip_ini, "r") as fd:
            for line in fd.readlines():
                line = line.split(';')[0].strip()
                if len(line) > 0:
                    hash_input += line
        return hashlib.sha512(hash_input.encode("utf-8")).hexdigest()
    finally:
        os.remove(final_comskip_ini)


def build_comskip_ini(comskip_ini, leaf_comskip_ini=None, video_path=None, input_info=None,
                      workdir=tempfile.gettempdir(), keep=False) -> str:
    """
    Build a final comskip.ini file from a base comskip.ini and customize based on video properties.
    :param comskip_ini:
    :param leaf_comskip_ini:
    :param video_path:
    :param input_info:
    :param workdir:
    :param keep: False to delete when other temp files are deleted, True to keep
    :return: file path
    """
    if input_info is None:
        input_info = common.find_input_info(video_path)
    elif video_path is None:
        video_path = input_info.get(common.K_FORMAT, {}).get("filename")

    if video_path:
        comskip_temp = os.path.join(workdir, f".~{os.path.basename(video_path)}.ini")
    else:
        fd, comskip_temp = tempfile.mkstemp(suffix=".ini", dir=workdir)
        os.close(fd)

    if not keep:
        common.TEMPFILENAMES.append(comskip_temp)

    duration_in_seconds = float(input_info[common.K_FORMAT]['duration'])
    ini_dir = os.path.dirname(os.path.abspath(comskip_ini))

    comskip_obj = configparser.ConfigParser()
    comskip_obj.read(comskip_ini)
    filenames = []

    # Time based modifications
    if duration_in_seconds < 35 * 60:
        filenames.append(os.path.join(ini_dir, 'comskip-30m.ini'))
    elif duration_in_seconds < 65 * 60:
        filenames.append(os.path.join(ini_dir, 'comskip-60m.ini'))
    elif duration_in_seconds < 95 * 60:
        filenames.append(os.path.join(ini_dir, 'comskip-90m.ini'))
    elif duration_in_seconds < 125 * 60:
        filenames.append(os.path.join(ini_dir, 'comskip-120m.ini'))
    elif duration_in_seconds < 155 * 60:
        filenames.append(os.path.join(ini_dir, 'comskip-150m.ini'))
    elif duration_in_seconds < 185 * 60:
        filenames.append(os.path.join(ini_dir, 'comskip-180m.ini'))
    else:
        filenames.append(os.path.join(ini_dir, 'comskip-INFm.ini'))

    # Resolution based
    height = common.get_video_height(input_info) or 480
    if height <= 480:
        filenames.append(os.path.join(ini_dir, 'comskip-480p.ini'))
    elif height <= 720:
        filenames.append(os.path.join(ini_dir, 'comskip-720p.ini'))
    elif height <= 1080:
        filenames.append(os.path.join(ini_dir, 'comskip-1080p.ini'))
    elif height <= 2160:
        # 4K
        filenames.append(os.path.join(ini_dir, 'comskip-4k.ini'))
    else:
        # 8K
        filenames.append(os.path.join(ini_dir, 'comskip-8k.ini'))

    # Video directory tree
    if video_path:
        path = video_path
        insert_idx = len(filenames)
        while True:
            p = os.path.dirname(os.path.abspath(path))
            if not p or p == path:
                break
            conf = os.path.join(p, 'comskip.ini')
            if os.path.isfile(conf):
                filenames.insert(insert_idx, conf)
            path = p

    # Video specific
    video_ini = common.replace_extension(video_path, 'comskip.ini')
    if os.path.isfile(video_ini):
        filenames.append(video_ini)

    if leaf_comskip_ini:
        filenames.append(leaf_comskip_ini)

    logger.debug(f"comskip config from {[comskip_ini] + filenames}")
    comskip_obj.read(filenames)

    # Necessary for the EDL file
    if not comskip_obj.has_section(INI_GROUP_OUTPUT_CONTROL):
        comskip_obj.add_section(INI_GROUP_OUTPUT_CONTROL)
    comskip_obj[INI_GROUP_OUTPUT_CONTROL]['output_edl'] = '1'

    # Necessary for the log file
    if not comskip_obj.has_section(INI_GROUP_MAIN_SETTINGS):
        comskip_obj.add_section(INI_GROUP_MAIN_SETTINGS)
    comskip_obj[INI_GROUP_MAIN_SETTINGS]['verbose'] = '10'

    with open(comskip_temp, "w") as f:
        comskip_obj.write(f, space_around_delimiters=False)

    return comskip_temp


def write_chapter_metadata(metafd, start_seconds, end_seconds, title, min_seconds=1) -> bool:
    logger.debug(f"write_chapter_metadata({start_seconds}, {end_seconds}, {title}, {min_seconds})")
    if start_seconds >= 0 and (end_seconds - start_seconds) >= min_seconds:
        metafd.write(f"""
[CHAPTER]
TIMEBASE=1/1000
START={int(start_seconds * 1000)}
END={int(end_seconds * 1000)}
title={title}
""")
        return True
    return False


def write_chapter_atom(mkvchapterfd, start_seconds, end_seconds, title, min_seconds=1) -> bool:
    logger.debug(f"write_chapter_atom({start_seconds}, {end_seconds}, {title}, {min_seconds})")
    if start_seconds >= 0 and (end_seconds - start_seconds) >= min_seconds:
        mkvchapterfd.write(f"""
    <ChapterAtom>
        <ChapterTimeStart>{common.s_to_ts(start_seconds)}</ChapterTimeStart>
        <ChapterTimeEnd>{common.s_to_ts(end_seconds)}</ChapterTimeEnd>
        <ChapterDisplay>
            <ChapterString>{title}</ChapterString>
            <ChapterLanguage>eng</ChapterLanguage>
        </ChapterDisplay>
    </ChapterAtom>\n""")
        return True
    return False


def get_comskip_hwassist_options() -> list[str]:
    """
    Get options for hardware acceleration for comskip if it's enabled in the configuration. It's safe to extend the
    command with the results of this method without checks.
    :return: hardware assist options, or empty list
    """
    options = []
    if common.get_global_config_boolean('comskip', 'hwaccel', fallback=False):
        options.append("--hwassist")
        if hwaccel.find_hwaccel_method() == hwaccel.HWAccelMethod.NVENC:
            options.append("--cuvid")
    return options


def comchap(*args, **kwargs):
    try:
        return do_comchap(*args, **kwargs)
    finally:
        common.finish()


def do_comchap(infile, outfile, edlfile=None, delete_edl=True, delete_meta=True, delete_log=True, delete_logo=True,
               delete_txt=True,
               delete_ini=True, verbose=False, workdir=None, comskipini=None, leaf_comskip_ini=None, modify_video=True,
               force=False, debug=False, backup_edl=False, use_csv=True, csvfile=None, input_info=None):
    ffmpeg = common.find_ffmpeg()
    mkvpropedit = common.find_mkvpropedit()
    if input_info is None:
        input_info = common.find_input_info(infile)
    duration = float(input_info[common.K_FORMAT]['duration'])

    if common.is_truthy(input_info.get(common.K_FORMAT, {}).get(common.K_TAGS, {}).get(common.K_COMSKIP_SKIP)) and not force:
        logger.info("%s: filter skipped due to %s property", infile, common.K_COMSKIP_SKIP)
        return 1

    chapters = input_info.get(common.K_CHAPTERS, []).copy()
    chapters.sort(key=lambda c: float(c['start_time']))
    has_chapters_from_source_media, chapters_commercials = common.has_chapters_from_source_media(input_info)
    if has_chapters_from_source_media:
        logger.info("%s: skipping because we found existing chapters from another source", infile)
        return 1

    if workdir is None:
        workdir = common.get_work_dir()

    if not comskipini:
        try:
            comskipini = find_comskip_ini()
        except OSError as e:
            logger.fatal(f"finding {comskipini}", exc_info=e)
            return 255

    logger.debug(f"comskipini at {comskipini}")

    if not os.path.isdir(workdir) or not os.access(workdir, os.W_OK):
        logger.fatal(f"Workdir not a writable directory at {workdir}")
        return 255

    outextension = outfile.split('.')[-1]
    infile_base = '.'.join(os.path.basename(infile).split('.')[0:-1])

    outfile_base = f".~{infile_base}"
    if edlfile is None:
        edlfile = os.path.join(os.path.dirname(infile) or '.', f"{infile_base}.edl")
    edlbakfile = common.replace_extension(edlfile, "bak.edl")
    hidden_edlfile = os.path.join(workdir, f"{outfile_base}.edl")
    metafile = os.path.join(workdir, f"{outfile_base}.ffmeta")
    mkvchapterfile = os.path.join(workdir, f"{outfile_base}.chapters.xml")
    tags_filename = os.path.join(workdir, f"{outfile_base}.tags.xml")
    logfile = os.path.join(workdir, f"{outfile_base}.log")
    logofile = os.path.join(workdir, f"{outfile_base}.logo.txt")
    txtfile = os.path.join(workdir, f"{outfile_base}.txt")
    if csvfile is None or not os.path.exists(csvfile):
        csvfile = os.path.join(workdir, f"{outfile_base}.csv")
    if delete_edl:
        common.TEMPFILENAMES.append(edlfile)
        common.TEMPFILENAMES.append(hidden_edlfile)
    if delete_meta:
        common.TEMPFILENAMES.append(metafile)
        common.TEMPFILENAMES.append(mkvchapterfile)
        common.TEMPFILENAMES.append(tags_filename)
    if delete_log:
        common.TEMPFILENAMES.append(logfile)
    if delete_logo and not os.path.exists(csvfile):
        # processing with CSV needs the logo file, don't delete if we have the csv
        common.TEMPFILENAMES.append(logofile)
    if delete_txt:
        common.TEMPFILENAMES.append(txtfile)

    comskipini_hash = compute_comskip_ini_hash(comskipini, leaf_comskip_ini=leaf_comskip_ini, video_path=infile,
                                               input_info=input_info,
                                               workdir=workdir if delete_ini else tempfile.gettempdir())
    current_comskip_hash = input_info.get(common.K_FORMAT, {}).get(common.K_TAGS, {}).get(
        common.K_COMSKIP_HASH)
    logger.debug("comskip.ini hash is %s, current hash is %s", comskipini_hash, current_comskip_hash)

    run_comskip = force
    if not os.path.isfile(edlfile):
        if backup_edl and not force and os.path.isfile(edlbakfile):
            shutil.copyfile(edlbakfile, edlfile)
        else:
            run_comskip = True
    elif backup_edl:
        if not os.path.isfile(edlbakfile):
            shutil.copyfile(edlfile, edlbakfile)

    if modify_video:
        # the current hash is only available if we're allowed to modify the video
        if not current_comskip_hash:
            run_comskip = True
        elif current_comskip_hash != comskipini_hash:
            run_comskip = True

    if run_comskip:
        comskip_temp = build_comskip_ini(comskipini, leaf_comskip_ini=leaf_comskip_ini, video_path=infile,
                                         input_info=input_info, workdir=workdir,
                                         keep=not delete_ini)

        comskip_command = []
        if debug:
            comskip_command.append(common.find_comskip_gui())
            comskip_command.append("-w")
        else:
            comskip_command.append(common.find_comskip())
            comskip_command.extend(get_comskip_hwassist_options())

        # check for csv and logo file which makes the process much faster
        if not os.path.exists(csvfile):
            csvfile = common.replace_extension(infile, 'csv')
        if not os.path.isfile(csvfile):
            csvfile = common.replace_extension(os.path.join(workdir, os.path.basename(infile)), 'csv')
        logofile = common.replace_extension(csvfile, 'logo.txt')
        if not os.path.isfile(logofile):
            logofile = common.replace_extension(os.path.join(workdir, os.path.basename(infile)), 'logo.txt')

        comskip_command.extend([f"--output={workdir}", f"--output-filename={outfile_base}",
                                f"--ini={comskip_temp}"])

        if use_csv and os.path.isfile(logofile):
            comskip_command.append(f"--logo={logofile}")

        if use_csv and os.path.isfile(csvfile):
            comskip_command.append(csvfile)
        else:
            comskip_command.append(infile)

        logger.debug(' '.join(comskip_command))
        ret = subprocess.run(comskip_command, check=False, capture_output=not verbose)
        if not os.path.isfile(hidden_edlfile):
            logger.fatal(f"Error running comskip. EDL File not found: {hidden_edlfile}")
            return ret.returncode
        try:
            if hidden_edlfile != edlfile:
                shutil.move(hidden_edlfile, edlfile)
        except PermissionError as e:
            # copying permissions may have failed, but the file may have been copied
            if not os.path.exists(edlfile):
                raise e
        if backup_edl:
            shutil.copyfile(edlfile, edlbakfile)

    if not modify_video:
        return 0

    if common.assert_not_transcoding(infile, exit=False) != 0:
        return 255

    tags = input_info[common.K_FORMAT].get(common.K_TAGS, {}).copy()
    if common.K_COMSKIP_SKIP in tags:
        del tags[common.K_COMSKIP_SKIP]
    tags[common.K_COMSKIP_HASH] = comskipini_hash
    video_title = input_info[common.K_FORMAT].get(common.K_STREAM_TITLE, None)
    if video_title:
        tags[common.K_STREAM_TITLE] = video_title
    common.write_mkv_tags(tags, tags_filename)

    edl_events = common.parse_edl_cuts(edlfile)

    # check if already applied by comparing existing "Commercial" chapters
    if len(edl_events) == len(chapters_commercials):
        com_match = True
        for i, val in enumerate(edl_events):
            if int(float(val.start)) != int(float(chapters_commercials[i]['start_time'])):
                com_match = False
                break
            elif int(float(val.end)) != int(float(chapters_commercials[i]['end_time'])):
                com_match = False
                break
        if com_match:
            logger.info(f"{infile} already has correct chapters")
            if infile.endswith(
                    ".mkv") and current_comskip_hash != comskipini_hash and infile == outfile and mkvpropedit:
                mkvpropedit_command = [mkvpropedit, infile, "--tags", f"global:{tags_filename}"]
                logger.debug(' '.join(mkvpropedit_command))
                subprocess.run(mkvpropedit_command, check=True, capture_output=not verbose)
            return 255

    start = 0
    chapter_num = 0
    commercial_num = 0
    hascommercials = False

    with open(metafile, "w") as metafd:
        metafd.write(";FFMETADATA1\n")
        with open(mkvchapterfile, "w") as mkvchapterfd:
            mkvchapterfd.write(f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE Chapters SYSTEM "matroskachapters.dtd">
<Chapters><EditionEntry>\n""")

            for edl_event in edl_events:
                end = edl_event.start
                startnext = edl_event.end
                hascommercials = True

                if end > start:
                    chapter_num += 1
                    write_chapter_metadata(metafd, start, end, f"Chapter {chapter_num}")
                    write_chapter_atom(mkvchapterfd, start, end, f"Chapter {chapter_num}")

                if startnext > end:
                    commercial_num += 1
                    write_chapter_metadata(metafd, end, startnext, f"Commercial {commercial_num}")
                    write_chapter_atom(mkvchapterfd, end, startnext, f"Commercial {commercial_num}")

                start = startnext

            if hascommercials:
                if duration > start:
                    chapter_num += 1
                    write_chapter_metadata(metafd, start, duration, f"Chapter {chapter_num}")
                    write_chapter_atom(mkvchapterfd, start, duration, f"Chapter {chapter_num}")

            mkvchapterfd.write(f"""</EditionEntry></Chapters>\n""")

    # need this outside of metafd and mkvchapterfd blocks to ensure it's flushed and closed
    if hascommercials:
        if infile.endswith(".mkv") and outfile.endswith(".mkv") and mkvpropedit:
            # use mkvpropedit to apply chapters to mkv files, far more efficient
            if infile != outfile:
                shutil.copyfile(infile, outfile)
            mkvpropedit_command = [mkvpropedit, outfile, "--chapters", mkvchapterfile, "--tags",
                                   f"global:{tags_filename}"]
            logger.debug(' '.join(mkvpropedit_command))
            subprocess.run(mkvpropedit_command, check=True, capture_output=not verbose)
        else:
            ffmpeg_command = [ffmpeg, "-loglevel", "error", "-hide_banner", "-nostdin", "-i", infile, "-i", metafile,
                              "-map_metadata", "0",
                              "-map_chapters", "1",
                              "-map", "0",
                              "-metadata", f"{common.K_COMSKIP_HASH}={comskipini_hash}",
                              "-codec", "copy", "-y"]
            tempoutfile = None
            if infile == outfile:
                tempfd, tempoutfile = tempfile.mkstemp(suffix='.' + outextension, prefix='.~',
                                                       dir=os.path.dirname(os.path.abspath(outfile)))
                os.close(tempfd)
                ffmpeg_command.append(tempoutfile)
            else:
                ffmpeg_command.append(outfile)
            logger.debug(' '.join(ffmpeg_command))
            subprocess.run(ffmpeg_command, check=True, capture_output=not verbose)
            if tempoutfile is not None:
                os.replace(tempoutfile, outfile)
        if infile != outfile:
            common.match_owner_and_perm(target_path=outfile, source_path=infile)
    else:
        # add K_COMSKIP_HASH marker
        if infile.endswith(".mkv") and infile == outfile and mkvpropedit:
            mkvpropedit_command = [mkvpropedit, outfile, "--tags", f"global:{tags_filename}"]
            logger.debug(' '.join(mkvpropedit_command))
            subprocess.run(mkvpropedit_command, check=True, capture_output=not verbose)

    return 0


def comchap_mark_skip(filename: str, workdir: str) -> int:
    """
    Mark a file to skip the commercial scanner.
    :return: 0 for success
    """
    # add tag to media file
    input_info = common.find_input_info(filename)
    tags = input_info.get(common.K_FORMAT, {}).get(common.K_TAGS, {}).copy()
    if not common.is_truthy(tags.get(common.K_COMSKIP_SKIP)):
        # add tag to media file
        mkvpropedit = common.find_mkvpropedit()
        tags[common.K_COMSKIP_SKIP] = 'true'
        for key in [common.K_COMSKIP_HASH]:
            if key in tags:
                del tags[key]
        tags_filename = os.path.join(workdir, f".~{'.'.join(os.path.basename(filename).split('.')[0:-1])}.tags.xml")
        common.TEMPFILENAMES.append(tags_filename)
        common.write_mkv_tags(tags, tags_filename)
        mkvpropedit_args = [mkvpropedit, filename, "--tags", f"global:{tags_filename}"]

        # remove commercial chapter markers
        if not common.has_chapters_from_source_media(input_info)[0]:
            # lack of filename means delete chapters
            mkvpropedit_args.extend(['--chapters', ''])

        logger.debug(common.array_as_command(mkvpropedit_args))
        subprocess.run(mkvpropedit_args)

    # remove commercial files
    filename_without_ext = '.'.join(filename.split('.')[0:-1])
    for ext in ['.comskip.ini', '.bak.edl', '.edl']:
        filename_to_remove = filename_without_ext + ext
        try:
            os.remove(filename_to_remove)
            logger.debug("Removed %s", filename_to_remove)
        except FileNotFoundError:
            pass

    return 0


def comchap_cli(argv):
    delete_edl = not common.get_global_config_boolean('general', 'keep_edl')
    backup_edl = False
    delete_meta = not common.get_global_config_boolean('general', 'keep_meta')
    delete_log = True
    delete_logo = True
    delete_txt = True
    delete_ini = True
    verbose = False
    workdir = common.get_work_dir()
    comskipini = None
    modify_video = True
    force = False
    debug = False
    mark_skip = False

    try:
        opts, args = getopt.getopt(list(argv), "fk",
                                   ["keep-edl", "only-edl", "keep-meta", "keep-log", "keep-ini", "keep-all", "verbose",
                                    "comskip-ini=", "debug", "work-dir=", "force", "backup-edl", "mark-skip"])
    except getopt.GetoptError:
        usage()
        return 255
    for opt, arg in opts:
        if opt == '--help':
            usage()
            return 255
        elif opt == "--keep-edl":
            delete_edl = False
        elif opt == "--backup-edl":
            backup_edl = True
        elif opt == "--only-edl":
            delete_edl = False
            modify_video = False
        elif opt == "--keep-meta":
            delete_meta = False
        elif opt == "--keep-log":
            delete_log = False
        elif opt == "--keep-ini":
            delete_ini = False
        elif opt in ["-k", "--keep-all"]:
            delete_edl = False
            delete_meta = False
            delete_log = False
            delete_logo = False
            delete_txt = False
            delete_ini = False
        elif opt == "--verbose":
            verbose = True
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt == "--debug":
            debug = True
        elif opt == "--comskip-ini":
            comskipini = arg
            if not os.access(comskipini, os.R_OK):
                logger.fatal(f"Cannot find comskip.ini at {comskipini}")
                sys.exit(255)
        elif opt == "--work-dir":
            workdir = arg
        elif opt in ["-f", "--force"]:
            force = True
        elif opt == "--mark-skip":
            mark_skip = True

    if not args:
        usage()
        return 255

    atexit.register(common.finish)

    return_code = 0
    for infile, outfile in common.generate_video_files(args):
        if mark_skip:
            this_file_return_code = comchap_mark_skip(outfile, workdir)
        else:
            this_file_return_code = comchap(infile, outfile, delete_edl=delete_edl, delete_meta=delete_meta,
                                            delete_log=delete_log, delete_logo=delete_logo, delete_txt=delete_txt,
                                            delete_ini=delete_ini, verbose=verbose, workdir=workdir,
                                            comskipini=comskipini, modify_video=modify_video, force=force, debug=debug,
                                            backup_edl=backup_edl)
        if this_file_return_code != 0 and return_code == 0:
            return_code = this_file_return_code

    return return_code


if __name__ == '__main__':
    os.nice(12)
    common.setup_cli()
    sys.exit(comchap_cli(sys.argv[1:]))
