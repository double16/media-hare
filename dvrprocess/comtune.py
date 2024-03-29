#!/usr/bin/env python3
import atexit
import configparser
import getopt
import hashlib
import logging
import os.path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from math import ceil
from multiprocessing import Pool
from statistics import stdev, mean
from typing import Union

import numpy
import pygad

import common
from comchap import comchap, build_comskip_ini, find_comskip_ini, get_expected_adjusted_duration, \
    INI_GROUP_MAIN_SETTINGS, INI_GROUP_MAIN_SCORING, INI_GROUP_GLOBAL_REMOVES, INI_GROUP_LOGO_FINDING, \
    INI_GROUP_LOGO_INTERPRETATION, INI_GROUP_VERSIONS, INI_ITEM_VERSIONS_VIDEO_STATS, INI_ITEM_VERSIONS_GAD_TUNING, \
    INI_GROUP_DETAILED_SETTINGS, get_comskip_hwassist_options
from common import tools, config, constants, edl_util, progress

CSV_SUFFIX_BLACKFRAME = "-blackframe"

logger = logging.getLogger(__name__)

COL_FRAME = 0
COL_BRIGHTNESS = 1
COL_UNIFORM = 4
COL_SOUND = 5
VERSION_VIDEO_STATS = "1"
VERSION_GAD_TUNING = "2"


class ComskipGene(object):
    def __init__(self, config: tuple, use_csv: bool, description: str, exclude_if_default, space, data_type,
                 default_value):
        """
        :param config: tuple of (group, item) for the comskip.ini file
        :param use_csv: True if the config can use the CSV to improve performance
        :param description: human readable description
        :param exclude_if_default: True to exclude the config item if the optimal value is the default
        :param space: The configuration value space
        :param data_type: The python data type
        :param default_value: The default value, expected to match the default comskip.ini or comskip executable default
        """
        self.config = config
        self.use_csv = use_csv
        self.description = description
        self.exclude_if_default = exclude_if_default
        self.data_type = data_type
        self.default_value = default_value
        if type(space) in [range, numpy.ndarray]:
            self.space = list(space)
        else:
            self.space = space

    def space_has_elements(self):
        """
        Check if the gene space has more than 1 element, i.e. we have options to try.
        :return: True if len(space) > 1
        """
        if type(self.space) in [list, range]:
            return len(self.space) > 1
        return False


GENES: list[ComskipGene] = [
    # Detection Methods
    # 	  1 - Uniform Frame
    # 	  2 - Logo
    # 	  4 - Scene Change
    # 	  8 - Resolution Change
    # 	 16 - Closed Captions
    # 	 32 - Aspect Ratio
    # 	 64 - Silence
    # 	128 - CutScenes
    # 	255 - USE ALL AVAILABLE
    #
    # 41 - uniform frame, resolution change, aspect ratio
    # 43 - uniform frame, logo, resolution change, aspect ratio (plex 1.25 value)
    # 47 - uniform frame, logo, scene change, resolution change, aspect ratio
    # 111 - uniform frame, logo, scene change, resolution change, aspect ratio, silence
    # 175 - uniform frame, logo, scene change, resolution change, aspect ratio, cut scenes
    # 237 - uniform frame, scene change, resolution change, aspect ratio, silence, cut scenes
    # 239 - uniform frame, logo, scene change, resolution change, aspect ratio, silence, cut scenes
    # 255 - everything
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'detect_method'), True, "", False, [41, 43, 47, 111, 237, 239, 255], int, 239),
    ComskipGene((INI_GROUP_LOGO_FINDING, 'logo_threshold'), False, "", True, [0.70, 0.75, 0.90], [float, 2],
                0.75),
    ComskipGene((INI_GROUP_LOGO_FINDING, 'logo_filter'), False, "", True, [0, 2, 4], int, 0),
    # ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'connect_blocks_with_logo'), True, "", True, [0, 1], int, 1),
    ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'min_black_frames_for_break'), True, "", True, [1, 3, 5], int, 1),
    # Requires user entry: ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'shrink_logo'), True, "", True, [0, 1, 3, 5], int, 5),
    # Requires user entry: ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'shrink_logo_tail'), True, "", True, [0, 1, 2, 3], int, 0),
    ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'before_logo'), True, "", False, [0, 999], int, 0),
    ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'after_logo'), True, "", False, [0, 999], int, 0),
    # Calculated: ComskipGene((INI_GROUP_MAIN_SETTINGS, 'max_avg_brightness'), False, "", True, range(15, 60, 5), int, 20),
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'max_volume'), True, "", False, range(250, 1000, 50), int, 500),
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'non_uniformity'), True, "", False, range(250, 1000, 50), int, 500),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'length_strict_modifier'), True, "", True, numpy.arange(2.0, 5.01, 0.5),
                [float, 2], 3.0),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'length_nonstrict_modifier'), True, "", True, numpy.arange(1.0, 2.01, 0.25),
                [float, 2], 1.5),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'combined_length_strict_modifier'), True, "", True,
                numpy.arange(1.5, 2.51, 0.25), [float, 2], 2.0),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'combined_length_nonstrict_modifier'), True, "", True,
                numpy.arange(1.0, 1.51, 0.25), [float, 2], 1.25),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'ar_wrong_modifier'), True, "", True, [2.0], [float, 2], 2.0),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'ac_wrong_modifier'), True, "", True, [1.0], [float, 2], 1.0),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'excessive_length_modifier'), True, "", True,
                numpy.arange(0.005, 0.0151, 0.005), [float, 3], 0.01),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'dark_block_modifier'), True, "", True, numpy.arange(0.2, 0.51, 0.1),
                [float, 2], 0.3),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'min_schange_modifier'), True, "", True, numpy.arange(0.25, 0.751, 0.05),
                [float, 2], 0.5),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'max_schange_modifier'), True, "", True, numpy.arange(1, 3.01, 0.5),
                [float, 2], 2.0),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'logo_present_modifier'), True, "", True, numpy.arange(0.005, 0.0151, 0.005),
                [float, 3], 0.01),
    # punish bitmask: 1=brightness, 2=uniformity 4=volume, 8=silence amount, 16=scene change rate
    ComskipGene((INI_GROUP_DETAILED_SETTINGS, 'punish'), True, "", True, [0, 4, 4+8, 16, 4+8+16], int, 0),
    ComskipGene((INI_GROUP_GLOBAL_REMOVES, 'delete_show_before_or_after_current'), True, "", False, [0, 1], int, 0),
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'disable_heuristics'), True, "", False, [0, 4], int, 0),
]


def usage():
    print(f"""
Automated tuning of comskip configurations. Considers show seasons as a group.

Usage: {sys.argv[0]} file | dir

-p, --processes=2
-t, --time-limit={config.get_global_config_option('background_limits', 'time_limit')}
    Limit runtime. Set to 0 for no limit.
-e, --expensive
    Tune parameters that are expensive to compute, i.e. require processing of video for each value combination.
--verbose
-n, --dry-run
--work-dir={config.get_work_dir()}
-f, --force
    Force re-running tuning algorithm even if tuning is present. Give this option twice to re-run comskip CSV.
""", file=sys.stderr)


def comtune(*args, **kwargs):
    try:
        return do_comtune(*args, **kwargs)
    finally:
        common.finish()


def do_comtune(infile, verbose=False, workdir=None, force=0, dry_run=False):
    if workdir is None:
        workdir = os.path.dirname(os.path.abspath(infile))
    comskip_ini = find_comskip_starter_ini()
    try:
        ensure_framearray(infile, os.path.basename(infile) + CSV_SUFFIX_BLACKFRAME, comskip_ini, workdir=workdir,
                          dry_run=dry_run, force=force > 1)
    except subprocess.CalledProcessError:
        return 255

    max_avg_brightness, max_volume, non_uniformity = compute_black_frame_tunings(infile, workdir)
    infile_base, csv_path, video_ini = paths(infile, workdir)
    write_ini(video_ini, max_avg_brightness=max_avg_brightness, max_volume=max_volume, non_uniformity=non_uniformity)
    comchap(infile, infile, force=True, delete_edl=False, workdir=workdir)

    return 0


def compute_black_frame_tunings(infile, workdir) -> tuple[Union[int, None], Union[int, None], Union[int, None]]:
    """
    Compute the tuning values for determining black frames.
    :return: max_avg_brightness, max_volume, non_uniformity
    """
    infile_base, csv_path, video_ini = paths(infile, workdir,
                                             infile_base=os.path.basename(infile) + CSV_SUFFIX_BLACKFRAME)
    if not os.path.exists(csv_path):
        return None, None, None
    data = numpy.genfromtxt(csv_path, skip_header=2, delimiter=',')
    logger.info(f"{infile}: Loaded data {len(data)} frames")

    brightness_histo = histogram(data, COL_BRIGHTNESS, 0, 1)
    max_avg_brightness = int(min(list(map(lambda e: e[0], brightness_histo.items())))) + 5
    logger.info(f"{infile}: max_avg_brightness = {max_avg_brightness}")

    uniform_frames = list(filter(lambda r: r[COL_BRIGHTNESS] < max_avg_brightness and r[COL_UNIFORM] < 5, data))
    logger.info(f"{infile}: {len(uniform_frames)} uniform frames")
    volume_histo = histogram(uniform_frames, COL_SOUND, 0, 1)
    # logger.info(f"{infile}: uniform_frames = {uniform_frames}")
    if len(volume_histo) > 0:
        silence_level = int(min(list(map(lambda e: e[0], volume_histo.items()))))
        max_volume = silence_level * 4
        if max_volume < 250 or max_volume > 1000:
            logger.warning(f"max_volume calculation ({max_volume}) outside of expected range [250,1000]")
            max_volume = None
    else:
        max_volume = None
    logger.info(f"{infile}: max_volume = {max_volume}")

    silent_frames = list(filter(lambda r: r[COL_BRIGHTNESS] < max_avg_brightness and r[COL_SOUND] < 100, data))
    uniformity_histo = histogram(silent_frames, COL_UNIFORM, -1, 1)
    if len(uniformity_histo) > 0:
        non_uniformity = int(min(list(map(lambda e: e[0], uniformity_histo.items())))) * 4
        if non_uniformity < 250 or non_uniformity > 1000:
            logger.warning(f"non_uniformity calculation ({non_uniformity}) outside of expected range [250,1000]")
            non_uniformity = None
    else:
        non_uniformity = None
    logger.info(f"{infile}: non_uniformity = {non_uniformity}")

    return max_avg_brightness, max_volume, non_uniformity


def write_ini(path, max_avg_brightness, max_volume, non_uniformity):
    config = configparser.ConfigParser()
    if os.path.isfile(path):
        config.read(path)
    if not config.has_section(INI_GROUP_MAIN_SETTINGS):
        config.add_section(INI_GROUP_MAIN_SETTINGS)

    if max_avg_brightness is not None:
        config[INI_GROUP_MAIN_SETTINGS]['max_avg_brightness'] = str(max_avg_brightness)
        config[INI_GROUP_MAIN_SETTINGS]['test_brightness'] = str(max_avg_brightness + 20)
        config[INI_GROUP_MAIN_SETTINGS]['max_brightness'] = str(max_avg_brightness + 40)
    else:
        config[INI_GROUP_MAIN_SETTINGS].pop('max_avg_brightness', None)
        config[INI_GROUP_MAIN_SETTINGS].pop('test_brightness', None)
        config[INI_GROUP_MAIN_SETTINGS].pop('max_brightness', None)

    if max_volume is not None:
        config[INI_GROUP_MAIN_SETTINGS]['max_volume'] = str(max_volume)
    else:
        # comskip will compute max_volume when set to 0
        config[INI_GROUP_MAIN_SETTINGS]['max_volume'] = '0'

    if non_uniformity is not None:
        config[INI_GROUP_MAIN_SETTINGS]['non_uniformity'] = str(non_uniformity)
    else:
        config[INI_GROUP_MAIN_SETTINGS].pop('non_uniformity', None)

    if not config.has_section(INI_GROUP_VERSIONS):
        config.add_section(INI_GROUP_VERSIONS)
    config.set(INI_GROUP_VERSIONS, INI_ITEM_VERSIONS_VIDEO_STATS, VERSION_VIDEO_STATS)

    with open(path, "w") as f:
        config.write(f, space_around_delimiters=False)
    common.match_owner_and_perm(target_path=path, source_path=os.path.dirname(os.path.abspath(path)))


def write_ini_from_solution(path, genes: list[ComskipGene], solution):
    config = configparser.ConfigParser()
    if os.path.isfile(path):
        config.read(path)
    for idx, val in enumerate(solution):
        p = genes[idx].config
        if not config.has_section(p[0]):
            config.add_section(p[0])
        if val is None:
            config.remove_option(p[0], p[1])
        else:
            config.set(p[0], p[1], str(val))

    if not config.has_section(INI_GROUP_VERSIONS):
        config.add_section(INI_GROUP_VERSIONS)
    config.set(INI_GROUP_VERSIONS, INI_ITEM_VERSIONS_GAD_TUNING, VERSION_GAD_TUNING)

    with open(path, "w") as f:
        config.write(f, space_around_delimiters=False)
    common.match_owner_and_perm(target_path=path, source_path=os.path.dirname(os.path.abspath(path)))


def is_tuned(path, workdir):
    ini_path = None
    if os.path.isfile(path):
        infile_base, csv_path, ini_path = paths(path, workdir)
    if os.path.isdir(path):
        ini_path = os.path.join(os.path.abspath(path), 'comskip.ini')
    logger.debug("is_tuned(%s) looking for config %s", path, ini_path)

    if ini_path is None:
        return False
    if not os.path.isfile(ini_path):
        logger.debug("%s not found", ini_path)
        return False

    config = configparser.ConfigParser()
    config.read(ini_path)

    if config.getint(INI_GROUP_VERSIONS, INI_ITEM_VERSIONS_VIDEO_STATS, fallback=1) < int(VERSION_VIDEO_STATS):
        return False

    if os.path.isdir(path) and \
            config.getint(INI_GROUP_VERSIONS, INI_ITEM_VERSIONS_GAD_TUNING, fallback=1) < int(VERSION_GAD_TUNING):
        return False

    tuned_settings = [
        (INI_GROUP_MAIN_SETTINGS, 'max_avg_brightness'),
        (INI_GROUP_MAIN_SETTINGS, 'test_brightness'),
        (INI_GROUP_MAIN_SETTINGS, 'max_brightness'),
        (INI_GROUP_MAIN_SETTINGS, 'max_volume'),
    ]

    for section, item in tuned_settings:
        if not config.has_option(section, item):
            logger.debug("%s.%s.%s not found", ini_path, section, item)
            return False

    return True


def solution_repl(genes: list[ComskipGene], solution):
    vals = []
    for idx, val in enumerate(solution):
        vals.append(f"{genes[idx].config[1]}={val}")
    return ','.join(vals)


def edl_tempfile(infile, workdir):
    tempdir = workdir if workdir else os.path.dirname(os.path.abspath(infile))
    return os.path.join(tempdir, f'.~{os.path.splitext(os.path.basename(infile))[0]}.edl')


def setup_gad(pool: Pool, files, workdir, dry_run=False, force=0, expensive_genes=False, check_compute=True,
              num_generations=0, comskip_defaults: configparser.ConfigParser = None) ->\
        (object, list, list, list, progress.progress):
    """
    Creates and returns a fitness function for comskip parameters for the given video files.
    :param pool:
    :param files:
    :param workdir:
    :param dry_run:
    :param force:
    :param expensive_genes: True to use genes that require generating the CSV from video for each solution
    :param check_compute: True to stop processing if compute is too high
    :param comskip_defaults: 
    :return: fitness_func, genes, gene_space, gene_type
    """

    # TODO: support locking genes, i.e. detect_method if we need to exclude methods we know are broken for the recording

    genes = list(filter(lambda g: g.space_has_elements() and (g.use_csv or expensive_genes), GENES))
    logger.debug("fitting for genes: %s", list(map(lambda e: e.config, genes)))
    gene_space = list(map(lambda g: g.space, genes))
    gene_type = list(map(lambda g: g.data_type, genes))

    season_dir = os.path.dirname(files[0])
    comskip_ini_path = os.path.join(season_dir, 'comskip.ini')
    framearray_results = []

    # ffprobe info for all videos
    video_infos = []
    # ffprobe info for videos that are uncut from DVR
    dvr_infos = []
    durations = []
    input_dirs = set()
    for file_path in files:
        video_info = common.find_input_info(file_path)
        if not video_info:
            continue
        video_infos.append(video_info)
        episode_count, duration, file_duration = common.episode_info(video_info)
        durations.append(duration)
        if episode_count == 1:
            if common.is_from_dvr(video_info):
                dvr_infos.append(video_info)
            else:
                logger.info("Not considering %s because it's not from DVR", file_path)
        else:
            logger.info("Not considering %s because it has %d episodes", file_path, episode_count)
        input_dirs.add(os.path.dirname(file_path))

    if len(dvr_infos) == 0:
        raise UserWarning("No files look like they have commercials")

    # Ignore special longer episodes, such as the pilot
    episode_common_duration = common.get_common_episode_duration(dvr_infos)
    logger.info("Target episode duration %s", common.seconds_to_timespec(episode_common_duration))
    dvr_infos = list(filter(lambda info: episode_common_duration == common.round_episode_duration(info), dvr_infos))
    dvr_durations = list(map(lambda info: common.episode_info(info)[1], dvr_infos))

    if len(dvr_infos) == 0:
        raise UserWarning("No files look like they have commercials")

    expected_adjusted_duration = get_expected_adjusted_duration(dvr_infos[0])
    logger.info(f"Expected adjusted duration: {common.seconds_to_timespec(expected_adjusted_duration)}, "
                f"mean duration {common.seconds_to_timespec(mean(dvr_durations))}")

    black_frame_tuning_done = False
    if os.path.isfile(comskip_ini_path):
        config = configparser.ConfigParser()
        config.read(comskip_ini_path)
        if 'max_avg_brightness' in config[INI_GROUP_MAIN_SETTINGS]:
            black_frame_tuning_done = True

    if not black_frame_tuning_done:
        # generate CSV using starter configuration to determine black frame tuning
        comskip_starter_ini = find_comskip_starter_ini()
        for video_info in dvr_infos:
            file_path = video_info[constants.K_FORMAT]['filename']
            try:
                framearray_results.append(
                    pool.apply_async(common.pool_apply_wrapper(ensure_framearray), (
                        file_path, os.path.basename(file_path) + CSV_SUFFIX_BLACKFRAME, comskip_starter_ini, workdir,
                        dry_run,
                        True)))
            except [subprocess.CalledProcessError, KeyboardInterrupt] as e:
                pool.terminate()
                raise e
        video_stats_progress = progress.progress('blackframe tuning', 0, len(framearray_results) - 1)
        video_stats_progress.progress(0)
        try:
            for result_idx, result in enumerate(framearray_results):
                if check_compute and common.should_stop_processing():
                    pool.terminate()
                    raise StopIteration('over loaded')
                try:
                    result.wait()
                    result.get()
                    video_stats_progress.progress(result_idx)
                except subprocess.CalledProcessError as e:
                    # generate with the files we have
                    pass
                except KeyboardInterrupt as e:
                    pool.terminate()
                    raise e
        finally:
            video_stats_progress.stop()
        framearray_results.clear()

        max_avg_brightness_list = []
        max_volume_list = []
        non_uniformity_list = []
        for video_info in dvr_infos:
            file_path = video_info[constants.K_FORMAT]['filename']
            max_avg_brightness, max_volume, non_uniformity = compute_black_frame_tunings(file_path, workdir)
            if max_avg_brightness:
                max_avg_brightness_list.append(max_avg_brightness)
            if max_volume:
                max_volume_list.append(max_volume)
            if non_uniformity:
                non_uniformity_list.append(non_uniformity)

        write_ini(comskip_ini_path,
                  max_avg_brightness=int(mean(max_avg_brightness_list)) if max_avg_brightness_list else None,
                  non_uniformity=int(mean(non_uniformity_list)) if non_uniformity_list else None,
                  max_volume=None
                  )

    # create fitness function
    comskip_fitness_ini_path = os.path.join(workdir, 'comskip-fitness-'
                                            + hashlib.sha512(
        ",".join(filter(lambda e: os.path.basename(e), files)).encode("utf-8")).hexdigest()
                                            + '.ini')
    shutil.copyfile(comskip_ini_path, comskip_fitness_ini_path)

    tuning_progress = progress.progress(f"{input_dirs.pop()} tuning", 0, num_generations)

    def f(gad: pygad.GA, solution, solution_idx):
        write_ini_from_solution(comskip_fitness_ini_path, genes, solution)
        logger.debug(f"Calculating fitness for {solution_repl(genes, solution)}")

        csv_configs = []
        csv_values = []
        for idx, val in enumerate(solution):
            if not genes[idx].use_csv:
                csv_values.append(val)
                csv_configs.append(genes[idx].config)
        if len(csv_values) > 0:
            csv_values_str = ",".join(map(lambda e: str(e), csv_values))
            csv_hash = hashlib.sha512(csv_values_str.encode("utf-8")).hexdigest()
            logger.debug("gene values for generating csv: %s = %s, hash = %s", csv_configs, csv_values_str, csv_hash)
            csv_suffix = "-" + csv_hash
        else:
            csv_suffix = "-fitness"

        results = []
        for video_info in dvr_infos:
            file_path = video_info[constants.K_FORMAT]['filename']
            csvfile = common.replace_extension(
                os.path.join(workdir, common.remove_extension(os.path.basename(file_path)) + csv_suffix),
                'csv')
            edlfile = edl_tempfile(file_path, workdir)
            if os.path.exists(edlfile):
                os.remove(edlfile)
            results.append(pool.apply_async(common.pool_apply_wrapper(csv_and_comchap_generate), (),
                                            {
                                                'file_path': file_path,
                                                'comskip_ini_path': comskip_ini_path,
                                                'comskip_fitness_ini_path': comskip_fitness_ini_path,
                                                'csvfile': csvfile,
                                                'workdir': workdir,
                                                'video_info': video_info,
                                                'edlfile': edlfile,
                                                'dry_run': dry_run,
                                                'force_csv_regen': (force > 1 or not black_frame_tuning_done)
                                            },
                                            error_callback=common.error_callback_dump))

        csv_config_d = dict(zip(csv_configs, csv_values))
        video_stats_progress = progress.progress(
            'video stats '+','.join(map(lambda k: f"{k[1]}={csv_config_d[k]}", csv_config_d.keys())),
            0, len(results) - 1)
        try:
            for result_idx, result in enumerate(results):
                if check_compute and common.should_stop_processing():
                    pool.terminate()
                    raise StopIteration('over loaded')
                try:
                    result.wait()
                    result.get()
                    video_stats_progress.progress(result_idx)
                except subprocess.CalledProcessError:
                    # generate fitness with the files we have
                    pass
                except KeyboardInterrupt as e:
                    pool.terminate()
                    os.remove(comskip_fitness_ini_path)
                    raise e
        finally:
            video_stats_progress.stop()

        os.remove(comskip_fitness_ini_path)

        adjusted_durations = []
        # if we want to ignore already cut files, iterate over dvr_infos instead of video_infos
        for video_info in video_infos:
            file_path = video_info[constants.K_FORMAT]['filename']
            episode_count, episode_duration, video_duration = common.episode_info(video_info)
            adjusted_duration = video_duration
            edl_path = edl_tempfile(file_path, workdir)
            if os.access(edl_path, os.R_OK):
                for event in edl_util.parse_edl_cuts(edl_path):
                    this_duration = (event.end - event.start)
                    adjusted_duration -= this_duration
            adjusted_durations.append(adjusted_duration / episode_count)

        count_of_non_defaults = 0
        for idx, gene in enumerate(genes):
            if str(solution[idx]) != str(gene.default_value):
                count_of_non_defaults += 1

        sigma = stdev(adjusted_durations)
        avg = mean(adjusted_durations)
        expected_adjusted_duration_diff = abs(expected_adjusted_duration - avg)
        logger.info(
            f"Fitness for {solution_repl(genes, solution)}\nis "
            f"σ{common.seconds_to_timespec(sigma)}, "
            f"duration {common.seconds_to_timespec(avg)}, "
            f"expected_adjusted_duration_diff = {common.seconds_to_timespec(expected_adjusted_duration_diff)}, "
            f"count_of_non_defaults = {count_of_non_defaults}"
        )

        tuning_progress.progress(gad.generations_completed)

        return fitness_value(sigma, expected_adjusted_duration_diff, count_of_non_defaults, episode_common_duration)

    return f, genes, gene_space, gene_type, tuning_progress


def csv_and_comchap_generate(file_path, comskip_ini_path, comskip_fitness_ini_path, csvfile, workdir, video_info,
                             edlfile, dry_run, force_csv_regen):
    if force_csv_regen:
        if os.path.exists(csvfile) and os.stat(csvfile).st_mtime < os.stat(comskip_ini_path).st_mtime:
            logger.debug("Removing old csv file %s", csvfile)
            os.remove(csvfile)
    ensure_framearray(file_path, common.remove_extension(os.path.basename(csvfile)), comskip_fitness_ini_path, workdir,
                      dry_run)
    if not os.path.exists(csvfile):
        raise Exception(f"{csvfile} was not generated")

    comchap(file_path, file_path,
            leaf_comskip_ini=comskip_fitness_ini_path,
            workdir=workdir,
            input_info=video_info,
            edlfile=edlfile,
            delete_edl=False,
            modify_video=False,
            use_csv=True,
            csvfile=csvfile,
            force=True)


def fitness_value(sigma: float, expected_adjusted_duration_diff: float, count_of_non_defaults: float,
                  episode_common_duration: int = 60):
    # sigma good values 0 - 120
    # expected_adjusted_duration_diff good values 0 - 240
    # count_of_non_defaults good values 0 - 16

    if episode_common_duration <= 30:
        duration_tolerance = 30.0
    else:
        duration_tolerance = 60.0

    return 1.1 / (sigma + 0.001) + \
           1.2 / max(0.001, abs(expected_adjusted_duration_diff) - duration_tolerance) + \
           1.0 / (count_of_non_defaults + 1000.0)


def paths(infile, workdir, infile_base=None):
    if infile_base is None:
        infile_base = common.remove_extension(os.path.basename(infile))
    csv_path = os.path.join(workdir, f"{infile_base}.csv")
    video_ini = os.path.join(os.path.dirname(os.path.abspath(infile)), f"{infile_base}.comskip.ini")
    return infile_base, csv_path, video_ini


def histogram(data, idx, min_value_excl, min_count_excl):
    h = {}
    for r in data:
        val = r[idx]
        if val in h:
            h[val] += 1
        else:
            h[val] = 1
    h2 = dict(filter(lambda e: e[0] > min_value_excl and e[1] > min_count_excl, h.items()))
    return h2


def ensure_framearray(infile, infile_base, comskip_ini, workdir, dry_run=False, force=False):
    infile_base, csv_path, _ = paths(infile, workdir, infile_base=infile_base)
    if not force and os.path.isfile(csv_path):
        return
    command = []
    # command.extend(["-v", "9"])
    command.extend(get_comskip_hwassist_options())
    command.extend(["--quiet", "--csvout",
                    f"--ini={comskip_ini}",
                    f"--output={workdir}", f"--output-filename={infile_base}", infile])
    logger.info(tools.comskip.array_as_command(command))
    if not dry_run:
        tools.comskip.run(command, check=True)


def get_comskip_starter_ini_sources():
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    return [f"{os.environ['HOME']}/.comskip-starter.ini",
            f"{script_dir}/comskip-starter.ini",
            f"/etc/comskip-starter.ini"]


def find_comskip_starter_ini():
    for f in get_comskip_starter_ini_sources():
        if os.access(f, os.R_OK):
            return f

    raise OSError(f"Cannot find comskip-starter.ini in any of {','.join(get_comskip_starter_ini_sources())}")


def tune_show(season_dir, pool, files, workdir, dry_run, force, expensive_genes=False, check_compute=True, processes=0):
    if len(files) < 5:
        logger.warning("too few video files %d to tune %s, need 5", len(files), season_dir)
        return

    logger.info("tuning show %s", season_dir)

    if not workdir:
        workdir = season_dir

    # We'll use the defaults to determine if we need to be specific about a property
    comskip_defaults = configparser.ConfigParser()
    comskip_defaults_file = build_comskip_ini(find_comskip_ini(), leaf_comskip_ini="comskip-none.ini",
                                              video_path=files[0], workdir=tempfile.gettempdir())
    comskip_defaults.read(comskip_defaults_file)
    os.remove(comskip_defaults_file)

    # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    num_generations = 50
    num_parents_mating = 6
    sol_per_pop = 12

    try:
        fitness_func, genes, gene_space, gene_type, tuning_progress = setup_gad(
            pool, files, workdir=workdir, dry_run=dry_run, force=force, comskip_defaults=comskip_defaults,
            expensive_genes=expensive_genes, check_compute=check_compute, num_generations=num_generations)
    except UserWarning as e:
        logger.warning(e.args[0])
        return

    # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=len(genes),
                           gene_space=gene_space,
                           gene_type=gene_type,
                           mutation_type="adaptive",
                           mutation_percent_genes=[25, 12],
                           parent_selection_type="sss",
                           save_best_solutions=True,
                           suppress_warnings=True)
    ga_instance.run()
    tuning_progress.stop()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    logger.info(
        "Parameters of the best solution : {solution}".format(solution=solution_repl(genes, solution)))
    logger.info("Fitness value of the best solution = {solution_fitness}".format(
        solution_fitness=(1.0 / solution_fitness)))
    logger.debug("Best solutions = {best}\n   fitness = {fitness}".format(best=ga_instance.best_solutions,
                                                                          fitness=ga_instance.best_solutions_fitness))
    solution = solution.copy()
    gene_values = []
    for idx in range(len(genes)):
        gene_values.append(set())
    for best_solution_idx, best_solution in enumerate(ga_instance.best_solutions):
        if ga_instance.best_solutions_fitness[best_solution_idx] != solution_fitness:
            continue
        for idx, val in enumerate(best_solution):
            gene_values[idx].add(str(val))
    for idx, val in enumerate(gene_values):
        config = genes[idx].config
        if len(val) > 1:
            config_default = comskip_defaults.get(config[0], config[1], fallback=None)
            logger.info(f"{config} did not contribute to fitness: {val} (default {config_default})")
            if config_default in [None, val, str(genes[idx].default_value)] and genes[idx].exclude_if_default:
                logger.info(f"{config} removing because the default value is part of the solution space")
                solution[idx] = None
            if type(genes[idx].space) is list and len(genes[idx].space) == len(val):
                logger.info(f"{config} removing because all gene space is part of the solution space")
                solution[idx] = None
        else:
            logger.info(f"{config} DID contribute to fitness: {val}")

    logger.info(
        "Parameters of the adjusted solution : {solution}".format(solution=solution_repl(genes, solution)))

    write_ini_from_solution(os.path.join(season_dir, 'comskip.ini'), genes, solution)

    return_code = common.ReturnCodeReducer()
    for filepath in files:
        pool.apply_async(common.pool_apply_wrapper(comchap), (filepath, filepath),
                         {'force': True,
                          'delete_edl': False,
                          'workdir': workdir},
                         callback=return_code.callback,
                         error_callback=common.error_callback_dump)


def comtune_cli(argv) -> int:
    # TODO: Use Plex API to get candidates to improve performance

    no_curses = False
    verbose = False
    workdir = config.get_work_dir()
    force = 0
    dry_run = False
    expensive_genes = False
    check_compute = True
    time_limit = int(config.get_global_config_time_seconds('background_limits', 'time_limit'))

    processes = config.get_global_config_int('background_limits', 'processes',
                                             fallback=max(1, int(common.core_count() / 2) - 1))

    try:
        opts, args = getopt.getopt(list(argv), "hfnep:t:",
                                   ["help", "verbose", "work-dir=", "force", "dry-run", "processes=", "time-limit=",
                                    "expensive", "no-curses"])
    except getopt.GetoptError:
        usage()
        return 255
    for opt, arg in opts:
        if opt in ['-h', '--help']:
            usage()
            return 255
        elif opt == "--verbose":
            verbose = True
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt == "--no-curses":
            no_curses = True
        elif opt == "--work-dir":
            workdir = arg
        elif opt in ["-f", "--force"]:
            force += 1
        elif opt in ["-n", "--dry-run"]:
            dry_run = True
            no_curses = True
        elif opt in ["-t", "--time-limit"]:
            time_limit = config.parse_seconds(arg)
        elif opt in ["-p", "--processes"]:
            if len(arg) > 0:
                processes = int(arg)
            check_compute = False
        elif opt in ["-e", "--expensive"]:
            expensive_genes = True

    if check_compute and not common.should_start_processing():
        print("not enough compute available", file=sys.stderr)
        return 255

    if not args:
        args = common.get_media_paths()

    common.cli_wrapper(comtune_cli_run, media_paths=args, verbose=verbose, workdir=workdir, force=force,
                       dry_run=dry_run, expensive_genes=expensive_genes, check_compute=check_compute,
                       time_limit=time_limit, processes=processes, no_curses=no_curses)


def comtune_cli_run(media_paths: list[str], verbose: bool, workdir, force: int, dry_run: bool, expensive_genes: bool,
                    check_compute: bool, time_limit: int, processes: int) -> int:
    logger.debug("work_dir is %s, processes is %s", workdir, processes)

    atexit.register(common.finish)

    time_start = time.time()
    time_progress = progress.progress("time limit", 0, time_limit)
    time_progress.renderer = common.s_to_ts

    return_code = common.ReturnCodeReducer()
    pool = Pool(processes=processes)

    def single_file_tune(f):
        if is_tuned(f, workdir) and force == 0:
            logger.info("%s already tuned, use --force to force tuning", f)
        elif not common.is_from_dvr(common.find_input_info(f)):
            logger.info("Skipping %s because it does not look like a DVR", f)
        else:
            pool.apply_async(common.pool_apply_wrapper(comtune), (f,),
                             {'verbose': verbose,
                              'workdir': workdir,
                              'force': force,
                              'dry_run': dry_run},
                             callback=return_code.callback,
                             error_callback=common.error_callback_dump)

    def should_stop() -> bool:
        if time_start is not None:
            duration = time.time() - time_start
            time_progress.progress(ceil(duration))
            if 0 < time_limit < duration:
                logger.info(
                    f"Exiting normally after processing {common.s_to_ts(int(duration))}, limit of {common.s_to_ts(time_limit)} reached")
                return True

        if check_compute and common.should_stop_processing():
            logger.info(f"INFO: not enough compute available")
            return True

        return False

    try:
        random.shuffle(media_paths)
        for media_path in media_paths:
            media_path = os.path.abspath(media_path)
            if os.path.isfile(media_path):
                filepath = media_path
                if common.is_video_file(filepath):
                    single_file_tune(filepath)
            else:
                for root, dirs, files in os.walk(media_path):
                    if should_stop():
                        pool.close()
                        return return_code.code()
                    random.shuffle(dirs)
                    random.shuffle(files)
                    video_files = common.filter_video_files(root, files)
                    if len(video_files) == 0:
                        continue
                    is_show = "Season" in root or ("Season" in video_files[0])
                    if is_show:
                        season_dir = os.path.dirname(video_files[0])
                        try:
                            if is_tuned(season_dir, workdir) and force == 0:
                                logger.info("%s already tuned, use --force to force tuning", season_dir)
                            else:
                                tune_show(season_dir=season_dir, pool=pool, files=video_files, workdir=workdir,
                                          dry_run=dry_run, force=force, expensive_genes=expensive_genes,
                                          check_compute=check_compute, processes=processes)
                        except KeyboardInterrupt as e:
                            raise e
                        except BaseException as e:
                            logger.error(f"Skipping show {season_dir}, uncaught exception: {str(e)}", exc_info=e)
                    else:
                        for filepath in video_files:
                            if should_stop():
                                pool.close()
                                return return_code.code()
                            single_file_tune(filepath)
            if should_stop():
                pool.close()
                return return_code.code()

        pool.close()
    except KeyboardInterrupt:
        logger.info("User interrupt, waiting for pool to finish")
        return_code.set_code(130)
        pool.terminate()
    finally:
        common.pool_join_with_timeout(pool)
        time_progress.stop()

    return return_code.code()


if __name__ == '__main__':
    sys.exit(comtune_cli(sys.argv[1:]))
