#!/usr/bin/env python3
import atexit
import configparser
import getopt
import logging
import os.path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from statistics import stdev, mean

import numpy
import pygad

import common
from comchap import comchap, build_comskip_ini, find_comskip_ini, get_expected_adjusted_duration, \
    INI_GROUP_MAIN_SETTINGS, INI_GROUP_MAIN_SCORING, INI_GROUP_GLOBAL_REMOVES, INI_GROUP_LOGO_FINDING, \
    INI_GROUP_LOGO_INTERPRETATION, INI_GROUP_VERSIONS, INI_ITEM_VERSIONS_VIDEO_STATS, INI_ITEM_VERSIONS_GAD_TUNING

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
    # 43 - uniform frame, logo, fuzzy logic, aspect ratio (plex 1.25 value)
    # 47 - uniform frame, logo, scene change, fuzzy logic, aspect ratio
    # 111 - uniform frame, logo, scene change, fuzzy logic, aspect ratio, silence
    # 175 - uniform frame, logo, scene change, fuzzy logic, aspect ratio, cut scenes
    # 239 - uniform frame, logo, scene change, fuzzy logic, aspect ratio, silence, cut scenes
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'detect_method'), True, "", False, [43, 47, 111, 239], int, 239),
    # TODO: for use_csv = False, create hash of name=value with use_csv == False and use in CSV filename to allow
    #       running the GAD for these properties
    ComskipGene((INI_GROUP_LOGO_FINDING, 'logo_threshold'), False, "", True, numpy.arange(0.70, 0.95, 0.5), [float, 2],
                0.70),
    ComskipGene((INI_GROUP_LOGO_FINDING, 'logo_filter'), False, "", True, [0, 2, 4], int, 0),
    # ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'connect_blocks_with_logo'), True, "", True, [0, 1], int, 1),
    ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'min_black_frames_for_break'), True, "", True, [1, 3, 5], int, 1),
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'max_avg_brightness'), False, "", True, range(15, 60, 5), int, 20),
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
    ComskipGene((INI_GROUP_GLOBAL_REMOVES, 'delete_show_before_or_after_current'), True, "", False, [0, 1], int, 0),
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'disable_heuristics'), True, "", False, [0, 4], int, 0),
]


def usage():
    print(f"""
Automated tuning of comskip configurations. Considers show seasons as a group.

Usage: {sys.argv[0]} file | dir

-p, --processes=2
-t, --time-limit={common.get_global_config_time_seconds('background_limits', 'time_limit')}
    Limit runtime. Set to 0 for no limit.
--verbose
-n, --dry-run
--work-dir=
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
        ensure_framearray(infile, comskip_ini, workdir=workdir, dry_run=dry_run, force=force > 1)
    except subprocess.CalledProcessError:
        return 255

    max_avg_brightness, max_volume, non_uniformity = compute_black_frame_tunings(infile, workdir)
    infile_base, csv_path, video_ini = paths(infile, workdir)
    write_ini(video_ini, max_avg_brightness=max_avg_brightness, max_volume=max_volume, non_uniformity=non_uniformity)
    comchap(infile, infile, force=True, delete_edl=False, workdir=workdir)

    return 0


def compute_black_frame_tunings(infile, workdir):
    """
    Compute the tuning values for determining black frames.
    :return: max_avg_brightness, max_volume, non_uniformity
    """
    infile_base, csv_path, video_ini = paths(infile, workdir)
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


def is_tuned(path):
    ini_path = None
    if os.path.isfile(path):
        infile_base, csv_path, ini_path = paths(path, os.path.dirname(os.path.abspath(path)))
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
        (INI_GROUP_MAIN_SETTINGS, 'non_uniformity'),
    ]

    for section, item in tuned_settings:
        if not config.has_option(section, item) or config.getint(section, item) <= 0:
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


def setup_gad(pool: Pool, files, workdir, dry_run=False, force=0,
              comskip_defaults: configparser.ConfigParser = None):
    """
    Creates and returns a fitness function for comskip parameters for the given video files.
    :param pool:
    :param files:
    :param workdir:
    :param dry_run:
    :param force:
    :param comskip_defaults: 
    :return: fitness_func, genes, gene_space, gene_type
    """

    genes = list(filter(lambda g: g.use_csv and g.space_has_elements(), GENES))
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
    for filepath in files:
        video_info = common.find_input_info(filepath)
        if not video_info:
            continue
        video_infos.append(video_info)
        episode_count, duration, file_duration = common.episode_info(video_info)
        durations.append(duration)
        if episode_count == 1:
            if common.is_from_dvr(video_info):
                dvr_infos.append(video_info)
            else:
                logger.info("Not considering %s because it's not from DVR", filepath)
        else:
            logger.info("Not considering %s because it has %d episodes", filepath, episode_count)

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
            filepath = video_info[common.K_FORMAT]['filename']
            try:
                framearray_results.append(
                    pool.apply_async(ensure_framearray, (filepath, comskip_starter_ini, workdir, dry_run, True)))
            except subprocess.CalledProcessError:
                return 255
        for result in framearray_results:
            result.wait()
        framearray_results.clear()

        max_avg_brightness_list = []
        max_volume_list = []
        non_uniformity_list = []
        for video_info in dvr_infos:
            filepath = video_info[common.K_FORMAT]['filename']
            max_avg_brightness, max_volume, non_uniformity = compute_black_frame_tunings(filepath, workdir)
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

    # regenerate CSV with black frame settings
    for video_info in dvr_infos:
        filepath = video_info[common.K_FORMAT]['filename']
        try:
            framearray_results.append(pool.apply_async(ensure_framearray, (
                filepath, comskip_ini_path, workdir, dry_run, force > 1 or not black_frame_tuning_done)))
        except subprocess.CalledProcessError:
            return 255
    for result in framearray_results:
        result.wait()
    framearray_results.clear()

    # create fitness function
    comskip_fitness_ini_path = os.path.join(workdir, 'comskip-fitness.ini')
    shutil.copyfile(comskip_ini_path, comskip_fitness_ini_path)

    def f(solution, solution_idx):
        write_ini_from_solution(comskip_fitness_ini_path, genes, solution)
        logger.debug(f"Calculating fitness for {solution_repl(genes, solution)}")

        results = []
        for video_info in dvr_infos:
            filepath = video_info[common.K_FORMAT]['filename']
            results.append(pool.apply_async(comchap, (filepath, filepath),
                                            {'leaf_comskip_ini': comskip_fitness_ini_path,
                                             'workdir': workdir,
                                             'input_info': video_info,
                                             'edlfile': edl_tempfile(filepath, workdir),
                                             'delete_edl': False,
                                             'modify_video': False,
                                             'use_csv': True,
                                             'force': True}, error_callback=common.error_callback_dump))
        for result in results:
            result.wait()

        adjusted_durations = []
        for video_info in video_infos:
            filepath = video_info[common.K_FORMAT]['filename']
            episode_count, episode_duration, video_duration = common.episode_info(video_info)
            adjusted_duration = video_duration
            edl_path = edl_tempfile(filepath, workdir)
            if os.access(edl_path, os.R_OK):
                for event in common.parse_edl_cuts(edl_path):
                    this_duration = (event.end - event.start)
                    adjusted_duration -= this_duration
            # if we want to ignore already cut files, iterate over dvr_infos instead of video_infos
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
        return fitness_value(sigma, expected_adjusted_duration_diff, count_of_non_defaults, episode_common_duration)

    return f, genes, gene_space, gene_type


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


def paths(infile, workdir):
    infile_base = common.remove_extension(os.path.basename(infile))
    csv_path = os.path.join(workdir, f"{infile_base}.csv")
    video_ini = os.path.join(workdir, f"{infile_base}.comskip.ini")
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


def ensure_framearray(infile, comskip_ini, workdir, dry_run=False, force=False):
    infile_base, csv_path, _ = paths(infile, workdir)
    if not force and os.path.isfile(csv_path):
        return
    comskip = common.find_comskip()
    command = [comskip, "--hwassist", "--quiet", "--csvout", f"--ini={comskip_ini}", f"--output={workdir}",
               f"--output-filename={infile_base}", infile]
    logger.info(common.array_as_command(command))
    if not dry_run:
        subprocess.run(command, check=True, capture_output=True)


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


# FIXME: check for stopping processing so we don't overload the machine
def tune_show(season_dir, pool, files, workdir, dry_run, force):
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

    try:
        fitness_func, genes, gene_space, gene_type = setup_gad(pool, files, workdir, dry_run, force, comskip_defaults)
    except UserWarning as e:
        logger.warning(e.args[0])
        return

    num_generations = 50
    num_parents_mating = 6
    sol_per_pop = 12
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
                           save_best_solutions=True)
    ga_instance.run()
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

    os.remove(os.path.join(workdir, 'comskip-fitness.ini'))
    write_ini_from_solution(os.path.join(season_dir, 'comskip.ini'), genes, solution)
    for filepath in files:
        comchap(filepath, filepath, force=True, delete_edl=False, workdir=workdir)


def comtune_cli(argv):
    verbose = False
    workdir = common.get_work_dir()
    force = 0
    dry_run = False
    check_compute = True
    time_limit = int(common.get_global_config_time_seconds('background_limits', 'time_limit'))

    processes = common.get_global_config_int('background_limits', 'processes',
                                             fallback=max(1, int(common.core_count() / 2) - 1))

    try:
        opts, args = getopt.getopt(list(argv), "hfnp:t:",
                                   ["help", "verbose", "work-dir=", "force", "dry-run", "processes=", "time-limit="])
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
        elif opt == "--work-dir":
            workdir = arg
        elif opt in ["-f", "--force"]:
            force += 1
        elif opt in ["-n", "--dry-run"]:
            dry_run = True
        elif opt in ["-t", "--time-limit"]:
            time_limit = common.parse_seconds(arg)
        elif opt in ["-p", "--processes"]:
            if len(arg) > 0:
                processes = int(arg)
            check_compute = False

    if check_compute and not common.should_start_processing():
        logger.warning(f"not enough compute available")
        return 255

    if not args:
        args = common.get_media_paths()

    atexit.register(common.finish)

    time_start = time.time()
    return_code = common.ReturnCodeReducer()
    pool = Pool(processes=processes)

    def single_file_tune(f):
        if is_tuned(f) and force == 0:
            logger.info("%s already tuned, use --force to force tuning", f)
        elif common.has_chapters_from_source_media(common.find_input_info(f))[0]:
            logger.info(f"Skipping {f} because we found existing chapters from another source")
        else:
            pool.apply_async(comtune, (f,),
                             {'verbose': verbose,
                              'workdir': workdir,
                              'force': force,
                              'dry_run': dry_run}, return_code.callback,
                             common.error_callback_dump)

    def should_stop() -> bool:
        if time_start is not None:
            duration = time.time() - time_start
            if 0 < time_limit < duration:
                logger.info(
                    f"Exiting normally after processing {int(duration)}s, limit of {time_limit}s reached")
                return True

        if check_compute and common.should_stop_processing():
            logger.info(f"INFO: not enough compute available")
            return True

        return False

    try:
        random.shuffle(args)
        for arg in args:
            arg = os.path.abspath(arg)
            if os.path.isfile(arg):
                filepath = arg
                if common.is_video_file(filepath):
                    single_file_tune(filepath)
            else:
                for root, dirs, files in os.walk(arg):
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
                            if is_tuned(season_dir) and force == 0:
                                logger.info("%s already tuned, use --force to force tuning", season_dir)
                            else:
                                tune_show(season_dir=season_dir, pool=pool, files=video_files, workdir=workdir,
                                          dry_run=dry_run, force=force)
                        except BaseException as e:
                            logger.error(f"Skipping show {season_dir}, uncaught exception", exc_info=e)
                    else:
                        for file in files:
                            if should_stop():
                                pool.close()
                                return return_code.code()
                            filepath = os.path.join(root, file)
                            if common.is_video_file(filepath):
                                single_file_tune(filepath)
            if should_stop():
                pool.close()
                return return_code.code()

        pool.close()
    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.join()

    return return_code.code()


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(comtune_cli(sys.argv[1:]))
