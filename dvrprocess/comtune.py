#!/usr/bin/env python3
import configparser
import copy
import getopt
import hashlib
import json
import logging

import cloudpickle
import math
import os.path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, Future
from functools import lru_cache
from itertools import product
from math import ceil
from multiprocessing import Pool
from statistics import stdev, mean, median
from typing import Union, Any

import numpy
import pygad

import common
from comchap import comchap, build_comskip_ini, find_comskip_ini, get_expected_adjusted_duration, \
    INI_GROUP_MAIN_SETTINGS, INI_GROUP_MAIN_SCORING, INI_GROUP_GLOBAL_REMOVES, INI_GROUP_LOGO_FINDING, \
    INI_GROUP_LOGO_INTERPRETATION, INI_GROUP_VERSIONS, INI_ITEM_VERSIONS_VIDEO_STATS, INI_ITEM_VERSIONS_GAD_TUNING, \
    INI_GROUP_DETAILED_SETTINGS, get_comskip_hwassist_options, INI_GROUP_INPUT_CORRECTION
from common import tools, config, constants, edl_util, progress

CSV_SUFFIX_BLACKFRAME = "-blackframe"

logger = logging.getLogger(__name__)

COL_FRAME = 0
COL_BRIGHTNESS = 1
COL_UNIFORM = 4
COL_SOUND = 5
VERSION_VIDEO_STATS = "1"
VERSION_GAD_TUNING = "3"
debug = False


class ComskipGene(object):
    def __init__(self, config: tuple[str, str], use_csv: bool, description: str, exclude_if_default, space, data_type,
                 default_value, experimental=False):
        """
        :param config: tuple of (group, item) for the comskip.ini file
        :param use_csv: True if the config can use the CSV to improve performance
        :param description: human-readable description
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
        self.experimental = experimental
        if type(space) in [range, numpy.ndarray]:
            self.space = list(space)
        else:
            self.space = space

    def __str__(self):
        return str(self.config)

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
    # 107 - uniform frame, logo, resolution change, aspect ratio, silence
    # 111 - uniform frame, logo, scene change, resolution change, aspect ratio, silence
    # 175 - uniform frame, logo, scene change, resolution change, aspect ratio, cut scenes
    # 237 - uniform frame, scene change, resolution change, aspect ratio, silence, cut scenes
    # 239 - uniform frame, logo, scene change, resolution change, aspect ratio, silence, cut scenes
    # 255 - everything
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'detect_method'), True,
                "the sum of the values for which kind of frames comskip will consider as possible cutpoints",
                False, [41, 43, 47, 107, 111, 175, 237, 239, 255], int, 239),
    ComskipGene((INI_GROUP_LOGO_FINDING, 'logo_threshold'), False,
                "A logo is search using a logo mask. The logo threshold determines how much of the logo mask must match the video.",
                True, [0.70, 0.75, 0.80, 0.90], [float, 2], 0.75),
    ComskipGene((INI_GROUP_LOGO_FINDING, 'logo_filter'), False,
                "With a very noisy logo you can use this setting to enable a temporal filter on the logo detection.",
                True, [0, 2, 4], int, 0),
    ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'connect_blocks_with_logo'), True,
                "When enabled all blocks that have logo at the cut-point between the blocks will be considered one block",
                False, [0, 1], int, 1),
    ComskipGene((INI_GROUP_LOGO_FINDING, 'aggressive_logo_rejection'), False,
                "Set to higher values when the spatial logo detection is difficult",
                False, [0, 1, 4], int, 0),
    ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'min_black_frames_for_break'), True,
                "",
                True, [1, 3, 5], int, 1),
    # Requires user entry: ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'shrink_logo'), True, "", True, [0, 1, 3, 5], int, 5),
    # Requires user entry: ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'shrink_logo_tail'), True, "", True, [0, 1, 2, 3], int, 0),
    ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'before_logo'), True,
                "Cutpoints can be inserted just before the logo appears. This is the amount of seconds to start a search for silence before the logo appears.",
                False, [0, 2, 999], int, 0),
    ComskipGene((INI_GROUP_LOGO_INTERPRETATION, 'after_logo'), True,
                "Cutpoints can be inserted just after the logo disappears. This is the amount of seconds to start a search for silence after the logo disappears.",
                False, [0, 2, 999], int, 0),
    # Calculated: ComskipGene((INI_GROUP_MAIN_SETTINGS, 'max_brightness'), False, "", True, range(15, 60, 5), int, 60),
    # Calculated: ComskipGene((INI_GROUP_MAIN_SETTINGS, 'test_brightness'), False, "", True, range(15, 60, 5), int, 40),
    # Calculated: ComskipGene((INI_GROUP_MAIN_SETTINGS, 'max_avg_brightness'), False, "", True, range(15, 60, 5), int, 25),
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'maxbright'), False,
                "Amount of pixels in a black frame allowed to be brighter than max_brightness",
                False, range(1, 100, 20), int, 1, experimental=True),
    ComskipGene((INI_GROUP_DETAILED_SETTINGS, 'brightness_jump'), True,
                "Any frame with a jump in average brightness compared to the previous frame is a candidate scene change cutpoint",
                True, [100, 200, 300, 500], int, 200, experimental=True),
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'max_volume'), True,
                "The maximum sound volume allowed at or around a black frame, volume_slip determines the allowed offset in frames between sound and video",
                False, [250, 400, 500, 600, 700, 800, 900, 1000], int, 500),
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'max_silence'), False,
                "The maximum sound volume allowed at or around a black frame, volume_slip determines the allowed offset in frames between sound and video",
                False, [80, 100, 200, 300], int, 100, experimental=True),
    ComskipGene((INI_GROUP_DETAILED_SETTINGS, 'min_silence'), False,
                "The minimum number of frames the volume has to be below the silence level to be regarded as a silence cutpoint",
                False, [8, 12, 20], int, 12, experimental=True),
    ComskipGene((INI_GROUP_INPUT_CORRECTION, 'volume_slip'), True,
                "Maximum number of frames the silence is allowed to be misaligned with a blackframe to be regarded as a cut-point. When the broadcast has transmission errors and bad PTS a value of up to 200 can be required. A higher value increases the chance of false positives on black frames.",
                True, [40, 100, 200, 300], int, 40),
    # Calculated:
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'non_uniformity'), False,
                "The maximum fraction of pixels that are allowed to have more than noise_level brightness difference from the average brightness of a frame to be regarded as a uniform frame",
                False, [250, 500, 750, 1000], int, 500, experimental=True),
    ComskipGene((INI_GROUP_DETAILED_SETTINGS, 'noise_level'), False,
                "The maximum deviation of the average brightness in a uniform frame that allows pixels not to be counted as non uniform",
                False, [3, 5, 8, 10], int, 5, experimental=True),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'length_strict_modifier'), True,
                "Used when the block adheres to some strict criteria for commercials",
                True, numpy.arange(2.0, 5.01, 0.5), [float, 2], 3.0),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'length_nonstrict_modifier'), True,
                "Used when the block adheres to some lesser used criteria for commercials",
                True, numpy.arange(1.0, 2.01, 0.25), [float, 2], 1.5),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'combined_length_strict_modifier'), True,
                "used when a short number of sequential blocks adhere to the strict criteria for commercials",
                True, numpy.arange(1.5, 2.51, 0.25), [float, 2], 2.0),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'combined_length_nonstrict_modifier'), True,
                "used when a short number of sequential blocks adhere to the lesser used criteria for commercials",
                True, numpy.arange(1.0, 1.51, 0.25), [float, 2], 1.25),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'ar_wrong_modifier'), True,
                "Used when the aspect ratio of a block is different from the dominant aspect ratio",
                True, [2.0, 1000.0], [float, 2], 2.0),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'ac_wrong_modifier'), True,
                "Used when the number of audio channels versus the dominant number of audio channels, 1.0 means inactive",
                True, [1.0], [float, 2], 1.0),
    # excessive_length_modifier: 0.005 ensures block is marked as show, 1.0 effectively disables
    ComskipGene((INI_GROUP_MAIN_SCORING, 'excessive_length_modifier'), True,
                "Used when the length of a block exceeds min_show_segment",
                True, [0.01, 1.0], [float, 3], 0.01),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'dark_block_modifier'), True,
                "Used when a block is darker then the average",
                True, numpy.arange(0.2, 0.51, 0.1), [float, 2], 0.3),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'min_schange_modifier'), True,
                "Used when a block has much less then average scene changes",
                True, numpy.arange(0.25, 0.751, 0.05), [float, 2], 0.5),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'max_schange_modifier'), True,
                "Used when a block has much more then average scene changes",
                True, numpy.arange(1, 3.01, 0.5), [float, 2], 2.0),
    ComskipGene((INI_GROUP_MAIN_SCORING, 'logo_present_modifier'), True,
                "Used when a block has logo or use in reverse when there is no logo",
                True, numpy.arange(0.005, 0.0151, 0.005), [float, 3], 0.01),
    # punish bitmask: 1=brightness, 2=uniformity 4=volume, 8=silence amount, 16=scene change rate
    ComskipGene((INI_GROUP_DETAILED_SETTINGS, 'punish'), True,
                "Set the bitmask of the average audio/video aspects to monitor",
                True, [0, 4, 4 + 8, 16, 4 + 8 + 16], int, 0),
    ComskipGene((INI_GROUP_DETAILED_SETTINGS, 'punish_threshold'), True,
                "When the average is punish_threshold times above the average then it will be punished",
                True, [1.0, 1.3, 1.5], [float, 1], 1.3),
    ComskipGene((INI_GROUP_DETAILED_SETTINGS, 'punish_modifier'), True,
                "Used to modify the score when the punish is above the threshold",
                True, [1, 2, 4], int, 2),
    ComskipGene((INI_GROUP_DETAILED_SETTINGS, 'punish_no_logo'), True,
                "Do not modify the score of a block because it has no logo",
                True, [0, 1], int, 1, experimental=False),
    ComskipGene((INI_GROUP_GLOBAL_REMOVES, 'delete_show_before_or_after_current'), True,
                "Any part of the show that comes before or after the actual show and is separated from the show by a small commercial block less then min_commercial_break is deleted when that part is shorter then added_recording (1) or the amount of seconds set (2 or more).",
                False, [0, 1], int, 0),
    #  (1) H1: Deletes short show blocks between two commercial blocks
    #  (2) H2: Deletes short show blocks before or after commercials for various reasons
    #  (4) H3: Deletes or adds short blocks based on Logo interpretation
    #  (8) H4: Adds short blocks because of various reasons
    # (16) H5: Deletes show block before the first or after the last commercial
    # (32) H6: Deletes too short or too long commercials and too short commercials at the start of the end of the recording
    #       Related settings
    #         max_commercialbreak=600
    #         min_commercialbreak=20
    #         min_commercial_break_at_start_or_end=39
    # (64) H7: Tries to find the start of the show and deletes the short part of the previous show.
    ComskipGene((INI_GROUP_MAIN_SETTINGS, 'disable_heuristics'), True,
                "Bit pattern for disabling heuristics",
                False, [0, 1, 2, 4, 8, 16, 20, 32, 64, 255], int, 0),
]

# Genes for 30 minute show
GENES_30: list[ComskipGene] = [
    ComskipGene((INI_GROUP_GLOBAL_REMOVES, 'added_recording'), True,
                "Number of minutes added to show content to fill up the 60 or 30 minute slot",
                False, [7, 4], int, 7),
]

# Genes for 60 minute show
GENES_60: list[ComskipGene] = [
    ComskipGene((INI_GROUP_GLOBAL_REMOVES, 'added_recording'), True,
                "Number of minutes added to show content to fill up the 60 or 30 minute slot",
                False, [14, 9], int, 14),
]


@lru_cache(maxsize=None)
def find_gene(section: str, name: str) -> ComskipGene:
    config = (section, name)
    for gene in GENES:
        if gene.config == config:
            return gene
    raise ValueError(f"Unknown gene: {config}")


GENE_DETECT_METHOD = find_gene(INI_GROUP_MAIN_SETTINGS, 'detect_method')

# initial solutions
GENE_INITIAL_SOLUTION_VALUES: list[dict[ComskipGene: list]] = [
    # all detect methods with defaults
    {
        GENE_DETECT_METHOD: [],
        find_gene(INI_GROUP_MAIN_SETTINGS, 'disable_heuristics'): [0, 4],
        find_gene(INI_GROUP_LOGO_FINDING, 'aggressive_logo_rejection'): [0, 1],
        find_gene(INI_GROUP_MAIN_SCORING, 'ar_wrong_modifier'): [2.0, 1000.0],
    },
    #
    # recommended configurations
    #
    # https://www.kaashoek.com/comskip/viewtopic.php?f=7&t=1741
    {
        GENE_DETECT_METHOD: [107],
        # Calculated, Expensive: find_gene(INI_GROUP_MAIN_SETTINGS, 'max_brightness'): [60],
        # Calculated, Expensive: find_gene(INI_GROUP_MAIN_SETTINGS, 'test_brightness'): [40],
        # Calculated, Expensive: find_gene(INI_GROUP_MAIN_SETTINGS, 'max_avg_brightness'): [25],
        # Default: Expensive? find_gene(INI_GROUP_MAIN_SETTINGS, 'maxbright'): [1],
        # Default: Expensive? find_gene(INI_GROUP_DETAILED_SETTINGS, 'brightness_jump'): [200],
        # Config, maybe gene?: find_gene(INI_GROUP_MAIN_SETTINGS, 'max_commercialbreak'): [600],
        # Config, maybe gene?: find_gene(INI_GROUP_MAIN_SETTINGS, 'min_commercialbreak'): [4],
        # Config, maybe gene?: find_gene(INI_GROUP_MAIN_SETTINGS, 'max_commercial_size'): [140],
        # Config, maybe gene?: find_gene(INI_GROUP_MAIN_SETTINGS, 'min_commercial_size'): [4],
        # Config, maybe gene?: find_gene(INI_GROUP_MAIN_SETTINGS, 'min_show_segment_length'): [250],
        find_gene(INI_GROUP_MAIN_SETTINGS, 'non_uniformity'): [500],
        find_gene(INI_GROUP_MAIN_SETTINGS, 'max_volume'): [500],
        find_gene(INI_GROUP_MAIN_SETTINGS, 'max_silence'): [100],
        find_gene(INI_GROUP_DETAILED_SETTINGS, 'min_silence'): [12],
        find_gene(INI_GROUP_DETAILED_SETTINGS, 'noise_level'): [5],
        find_gene(INI_GROUP_DETAILED_SETTINGS, 'punish'): [1],
        find_gene(INI_GROUP_DETAILED_SETTINGS, 'punish_threshold'): [1.3],
        find_gene(INI_GROUP_DETAILED_SETTINGS, 'punish_modifier'): [4],
        find_gene(INI_GROUP_LOGO_FINDING, 'logo_threshold'): [0.80],
        find_gene(INI_GROUP_LOGO_INTERPRETATION, 'min_black_frames_for_break'): [1],
        find_gene(INI_GROUP_DETAILED_SETTINGS, 'punish_no_logo'): [1],
        find_gene(INI_GROUP_LOGO_FINDING, 'aggressive_logo_rejection'): [0],
        find_gene(INI_GROUP_LOGO_FINDING, 'logo_filter'): [0],
        # Arbitrary: find_gene(INI_GROUP_MAIN_SETTINGS, 'delete_show_after_last_commercial'): [0],
        # Arbitrary: find_gene(INI_GROUP_MAIN_SETTINGS, 'delete_show_before_first_commercial'): [0],
        # Arbitrary: find_gene(INI_GROUP_MAIN_SETTINGS, 'delete_show_before_or_after_current'): [0],
        # Arbitrary: find_gene(INI_GROUP_MAIN_SETTINGS, 'delete_block_after_commercial'): [0],
        # Arbitrary: find_gene(INI_GROUP_MAIN_SETTINGS, 'remove_before'): [0],
        # Arbitrary: find_gene(INI_GROUP_MAIN_SETTINGS, 'remove_after'): [0],
        # Arbitrary: find_gene(INI_GROUP_MAIN_SETTINGS, 'shrink_logo'): [5],
        find_gene(INI_GROUP_LOGO_INTERPRETATION, 'after_logo'): [0],
        find_gene(INI_GROUP_LOGO_INTERPRETATION, 'before_logo'): [0],
        find_gene(INI_GROUP_MAIN_SETTINGS, 'disable_heuristics'): [4],
    },
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
    return do_comtune(*args, **kwargs)


def do_comtune(infile, verbose=False, workdir=None, force=0, dry_run=False):
    # TODO: use pypad with fitness function that fits for sigma on commercial lengths and sigma on show lengths
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
    comchap(infile, infile, force=True, delete_edl=False, backup_edl=True, workdir=workdir)

    return 0


def compute_black_frame_tunings(infile, workdir) -> tuple[Union[int, None], Union[int, None], Union[int, None]]:
    """
    Compute the tuning values for determining black frames.
    # https://www.kaashoek.com/files/tuning.htm
    :return: max_avg_brightness, max_volume, non_uniformity
    """
    infile_base, csv_path, video_ini = paths(infile, workdir,
                                             infile_base=os.path.basename(infile) + CSV_SUFFIX_BLACKFRAME)
    if not os.path.exists(csv_path):
        return None, None, None
    data = numpy.genfromtxt(csv_path, skip_header=2, delimiter=',')
    logger.info(f"{infile}: Loaded data {len(data)} frames")

    brightness_histo = histogram(data, COL_BRIGHTNESS, 0, 10)
    if brightness_histo:
        max_avg_brightness = int(min(map(lambda e: e[0], brightness_histo.items()))) + 5
    else:
        max_avg_brightness = 19  # from comskip-defaults.ini
    logger.info(f"{infile}: max_avg_brightness = {max_avg_brightness}")

    # uniform_frames = list(filter(lambda r: r[COL_BRIGHTNESS] < max_avg_brightness and r[COL_UNIFORM] < 5, data))
    # logger.info(f"{infile}: {len(uniform_frames)} uniform frames")
    volume_histo = histogram(data, COL_SOUND, 9, 10)
    # logger.info(f"{infile}: uniform_frames = {uniform_frames}")
    if len(volume_histo) > 0:
        silence_level = int(min(map(lambda e: e[0], volume_histo.items())))
        max_volume = max(250, min(1000, silence_level * 4))
    else:
        max_volume = None
    logger.info(f"{infile}: max_volume = {max_volume}")

    silent_frames = list(filter(lambda r: r[COL_BRIGHTNESS] < max_avg_brightness, data))
    uniformity_histo = histogram(silent_frames, COL_UNIFORM, -1, 5)
    if len(uniformity_histo) > 0:
        non_uniformity = int(min(map(lambda e: e[0], uniformity_histo.items()))) * 4
        non_uniformity = max(250, non_uniformity)
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


def write_ini_from_solution(path, genes: list[ComskipGene], solution: list, write_complete_config=False,
                            comskip_defaults: configparser.ConfigParser = None):
    if write_complete_config and comskip_defaults:
        config = copy.deepcopy(comskip_defaults)
    else:
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


def setup_gad(process_pool: Pool, thread_pool: ThreadPoolExecutor, files, workdir, dry_run=False, force=0,
              expensive_genes=False, check_compute=True,
              num_generations=0, comskip_defaults: configparser.ConfigParser = None, experimental=False,
              file_sample_size=None) -> \
        (object, list, list, list, progress.progress, str):
    """
    Creates and returns a fitness function for comskip parameters for the given video files.
    :param process_pool:
    :param thread_pool:
    :param files:
    :param workdir:
    :param dry_run:
    :param force:
    :param expensive_genes: True to use genes that require generating the CSV from video for each solution
    :param check_compute: True to stop processing if compute is too high
    :param comskip_defaults:
    :param file_sample_size: if >0 only include this many files in the algorithm to decrease total time
    :return: fitness_func, genes, gene_space, gene_type, fitness_json_path
    """

    season_dir = os.path.dirname(files[0])
    comskip_ini_path = os.path.join(season_dir, 'comskip.ini')
    framearray_results: list[Future] = []

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

    expected_adjusted_duration_default = get_expected_adjusted_duration(dvr_infos[0])
    logger.info(f"Mean duration {common.seconds_to_timespec(mean(dvr_durations))}")

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
                    thread_pool.submit(ensure_framearray,
                                       file_path, os.path.basename(file_path) + CSV_SUFFIX_BLACKFRAME,
                                       comskip_starter_ini, workdir,
                                       dry_run,
                                       True))
            except [subprocess.CalledProcessError, KeyboardInterrupt] as e:
                thread_pool.shutdown(cancel_futures=True)
                raise e
        video_stats_progress = progress.progress('blackframe tuning', 0, len(framearray_results) - 1)
        video_stats_progress.progress(0)
        try:
            for result_idx, result in enumerate(framearray_results):
                if check_compute and common.should_stop_processing():
                    thread_pool.shutdown(cancel_futures=True)
                    raise StopIteration('over loaded')
                try:
                    result.result()
                    video_stats_progress.progress(result_idx)
                except subprocess.CalledProcessError:
                    # generate with the files we have
                    pass
                except KeyboardInterrupt as e:
                    thread_pool.shutdown(cancel_futures=True)
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
                  max_avg_brightness=int(median(max_avg_brightness_list)) if max_avg_brightness_list else None,
                  non_uniformity=int(median(non_uniformity_list)) if non_uniformity_list else None,
                  max_volume=int(median(max_volume_list)) if non_uniformity_list else None,
                  )

    # construct list of genes
    # TODO: support locking genes, i.e. detect_method if we need to exclude methods we know are broken for the recording
    genes_all = GENES.copy()
    if episode_common_duration == 30 * 60:
        genes_all.extend(GENES_30)
    elif episode_common_duration == 60 * 60:
        genes_all.extend(GENES_60)
    genes = list(
        filter(lambda g: (experimental or not g.experimental) and g.space_has_elements() and (
                g.use_csv or expensive_genes), genes_all))
    permutations = math.prod(map(lambda g: len(g.space), genes))
    logger.info("fitting for genes: %s, permutations %d", list(map(lambda e: e.config, genes)), permutations)
    gene_space = list(map(lambda g: g.space, genes))
    gene_type = list(map(lambda g: g.data_type, genes))
    added_recording_gene_idx = -1
    for idx, g in enumerate(genes):
        if g.config[1] == 'added_recording':
            added_recording_gene_idx = idx
            break

    # create fitness function
    filename_hash = hashlib.sha512(",".join(filter(lambda e: os.path.basename(e), files)).encode("utf-8")).hexdigest()
    comskip_fitness_ini_path = os.path.join(workdir, 'comskip-fitness-' + filename_hash + '.ini')
    shutil.copyfile(comskip_ini_path, comskip_fitness_ini_path)

    fitness_json_path = os.path.join(workdir, 'comskip-fitness-' + filename_hash + '.json')
    if os.path.exists(fitness_json_path):
        os.remove(fitness_json_path)

    tuning_progress = progress.progress(f"{input_dirs.pop()} tuning", 0, num_generations + 1)

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

        dvr_infos_sample = dvr_infos.copy()
        if file_sample_size and file_sample_size > 0:
            random.shuffle(dvr_infos_sample)
            dvr_infos_sample = dvr_infos_sample[:file_sample_size]

        results: list[Future] = []
        for video_info in dvr_infos_sample:
            file_path = video_info[constants.K_FORMAT]['filename']
            csvfile = common.replace_extension(
                os.path.join(workdir, common.remove_extension(os.path.basename(file_path)) + csv_suffix),
                'csv')
            edlfile = edl_tempfile(file_path, workdir)
            if os.path.exists(edlfile):
                os.remove(edlfile)
            results.append(thread_pool.submit(csv_and_comchap_generate,
                                              file_path=file_path,
                                              comskip_ini_path=comskip_ini_path,
                                              comskip_fitness_ini_path=comskip_fitness_ini_path,
                                              csvfile=csvfile,
                                              workdir=workdir,
                                              video_info=video_info,
                                              edlfile=edlfile,
                                              dry_run=dry_run,
                                              force_csv_regen=(force > 1 or not black_frame_tuning_done)
                                              ))

        for result_idx, result in enumerate(results):
            if check_compute and common.should_stop_processing():
                thread_pool.shutdown(cancel_futures=True)
                raise StopIteration('over loaded')
            try:
                result.result()
            except subprocess.CalledProcessError:
                # generate fitness with the files we have
                pass
            except KeyboardInterrupt as e:
                thread_pool.shutdown(cancel_futures=True)
                os.remove(comskip_fitness_ini_path)
                raise e

        # added_recording may be a gene, so we need to calculate it for each run
        if added_recording_gene_idx >= 0:
            expected_adjusted_duration = common.round_episode_duration(dvr_infos_sample[0]) - (
                        (int(solution[added_recording_gene_idx]) + 1) * 60.0)
        else:
            expected_adjusted_duration = expected_adjusted_duration_default

        os.remove(comskip_fitness_ini_path)

        adjusted_durations = []
        commercial_breaks: list[list[edl_util.EdlEvent]] = []
        for video_info in dvr_infos_sample:
            file_path = video_info[constants.K_FORMAT]['filename']
            episode_count, episode_duration, video_duration = common.episode_info(video_info)
            adjusted_duration = video_duration
            edl_path = edl_tempfile(file_path, workdir)
            if os.access(edl_path, os.R_OK):
                _, this_commercial_breaks, duration_adjustment = edl_util.parse_commercials(edl_path, video_duration,
                                                                                            True, 0)
                adjusted_duration -= duration_adjustment
                commercial_breaks.append(this_commercial_breaks)
            adjusted_durations.append(adjusted_duration / episode_count)

        count_of_non_defaults = 0
        for idx, gene in enumerate(genes):
            if str(solution[idx]) != str(gene.default_value):
                count_of_non_defaults += 1

        sigma = stdev(adjusted_durations)
        avg = mean(adjusted_durations)
        expected_adjusted_duration_diff = abs(expected_adjusted_duration - avg)
        logger.debug(
            f"Fitness for {solution_repl(genes, solution)}\nis "
            f"σ{common.seconds_to_timespec(sigma)}, "
            f"duration {common.seconds_to_timespec(avg)}, "
            f"expected_adjusted_duration {common.seconds_to_timespec(expected_adjusted_duration)}, "
            f"expected_adjusted_duration_diff = {common.seconds_to_timespec(expected_adjusted_duration_diff)}, "
            f"count_of_non_defaults = {count_of_non_defaults}"
        )

        tuning_progress.progress(gad.generations_completed)

        return fitness_value(sigma, expected_adjusted_duration_diff, count_of_non_defaults, episode_common_duration,
                             commercial_breaks, fitness_json_path=fitness_json_path)

    return f, genes, gene_space, gene_type, tuning_progress, fitness_json_path


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
                  episode_common_duration: int = 60,
                  commercial_breaks: list[list[edl_util.EdlEvent]] = None,
                  fitness_json_path=None):
    if episode_common_duration <= 30:
        duration_tolerance = 30.0
    else:
        duration_tolerance = 30.0
        # duration_tolerance = 60.0

    commercial_break_score = 1000000  # closer to 0 is better
    aligned_commercial_breaks = None
    if commercial_breaks is not None:
        # we care about removing commercials
        if len([item for sublist in commercial_breaks for item in sublist]) > 0:
            aligned_commercial_breaks, commercial_break_score, _ = edl_util.align_commercial_breaks(commercial_breaks)
            if commercial_break_score < 1:
                logger.warning("Commercial break score < 1: %f", commercial_break_score)
        else:
            # we care about removing commercials, so do not consider solutions that found no commercials
            logger.debug("No commercial breaks available for scoring")
            return -9999

    result = 0

    # sigma good values 0 - 120
    result += 1.0 * (1000 - sigma / 1.104)

    # expected_adjusted_duration_diff good values 0 - 240
    # If the numerator is too great, less ideal results occur to fit the expected duration
    # If too less, sigma and commercial_break_score converge to cutting nothing or far too much
    result += 5.0 * (1000 - max(1.0, abs(expected_adjusted_duration_diff) - duration_tolerance) / 1.4)

    # commercial_break_score, good values 0 - 800
    if commercial_break_score < 1000000:
        result += 4.0 * (1000 - commercial_break_score / 3)

    if fitness_json_path:
        with open(fitness_json_path, "a") as f:
            acb_primitive = list(map(lambda breaks1: list(
                map(lambda event: None if event is None else [event.start, event.end], breaks1)),
                                     aligned_commercial_breaks))
            f.write(json.dumps(
                {
                    "sigma": sigma,
                    "expected_adjusted_duration_diff": expected_adjusted_duration_diff,
                    "duration_tolerance": duration_tolerance,
                    "count_of_non_defaults": count_of_non_defaults,
                    "commercial_break_score": commercial_break_score,
                    "aligned_commercial_breaks": acb_primitive,
                    "result": result,
                }
            ))
            f.write('\n')

    return result


def paths(infile, workdir, infile_base=None):
    if infile_base is None:
        infile_base = common.remove_extension(os.path.basename(infile))
    csv_path = os.path.join(workdir, f"{infile_base}.csv")
    video_ini = os.path.join(os.path.dirname(os.path.abspath(infile)), f"{infile_base}.comskip.ini")
    return infile_base, csv_path, video_ini


def histogram(data, idx: int, min_value_excl: int, min_count_excl: int):
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


@lru_cache(maxsize=None)
def get_comskip_starter_ini_sources():
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    return [f"{os.environ['HOME']}/.comskip-starter.ini",
            f"{script_dir}/comskip-starter.ini",
            "/etc/comskip-starter.ini"]


@lru_cache(maxsize=None)
def find_comskip_starter_ini():
    for f in get_comskip_starter_ini_sources():
        if os.access(f, os.R_OK):
            return f

    raise OSError(f"Cannot find comskip-starter.ini in any of {','.join(get_comskip_starter_ini_sources())}")


GA_INSTANCE_ATTR_SAVE = [
    'best_solutions',
    'best_solutions_fitness',
    'solutions',
    'solutions_fitness',
    'last_generation_fitness',
    'last_generation_parents',
    'last_generation_offspring_crossover',
    'last_generation_offspring_mutation',
    'previous_generation_fitness',
    'last_generation_elitism',
    'last_generation_elitism_indices',
    'pareto_fronts',
]


def ga_instance_save(ga_instance: pygad.GA, filename):
    gad_state = {
        'num_generations': ga_instance.num_generations,
        'generations_completed': ga_instance.generations_completed,
        'population': ga_instance.population,
    }

    for attr_name in GA_INSTANCE_ATTR_SAVE:
        gad_state[attr_name] = getattr(ga_instance, attr_name)

    with open(filename, 'wb') as file:
        cloudpickle.dump(gad_state, file)

    return None


def generate_initial_solutions(genes: list[ComskipGene], values_in: dict[ComskipGene, list]) -> list[list]:
    # remove values that are not in the list of genes
    values: dict[ComskipGene, list] = dict()
    for gene, gene_values in values_in.items():
        if gene in genes:
            if not gene_values:
                values[gene] = gene.space
            else:
                values[gene] = gene_values

    permutation_values: list[list] = list()
    for i in range(len(genes)):
        permutation_values.append(list())

    # Add initial values
    for gene, gene_values in values.items():
        idx = genes.index(gene)
        permutation_values[idx] = gene_values
    # Populate the remaining genes with defaults
    for idx, values in enumerate(permutation_values):
        if len(values) == 0:
            permutation_values[idx].append(genes[idx].default_value)

    permutation_tuples = list(product(*permutation_values))
    solutions = [list(perm) for perm in permutation_tuples]
    return solutions


def initial_solution_values_from_ini(path) -> dict[ComskipGene, list]:
    result = {}
    if not os.path.exists(path):
        return result

    config = configparser.ConfigParser()
    config.read(path)

    for section in config.sections():
        for item, value in config.items(section):
            try:
                gene = find_gene(section, item)
                try:
                    result[gene] = [int(value)]
                except ValueError:
                    try:
                        result[gene] = [float(value)]
                    except ValueError:
                        result[gene] = [value]
            except ValueError:
                # no gene defined for this item
                pass

    return result


def tune_show(season_dir, process_pool: Pool, files, workdir, dry_run, force, expensive_genes=False, check_compute=True,
              processes=0, experimental=False):
    if len(files) < 5:
        logger.warning("too few video files %d to tune %s, need 5", len(files), season_dir)
        return

    logger.info("tuning show %s", season_dir)

    if not workdir:
        workdir = season_dir

    # We'll use the defaults to determine if we need to be specific about a property
    comskip_defaults = configparser.ConfigParser()
    comskip_defaults_file = build_comskip_ini(find_comskip_ini(), leaf_comskip_ini="comskip-none.ini",
                                              video_path=files[0], workdir=tempfile.gettempdir(), log_file=False)
    comskip_defaults.read(comskip_defaults_file)
    os.remove(comskip_defaults_file)

    target_comskip_ini = os.path.join(season_dir, 'comskip.ini')

    gad_name_parts = ['gad']
    if expensive_genes:
        gad_name_parts.append('expensive_genes')
    if experimental:
        gad_name_parts.append('experimental')
    gad_state_filename = os.path.join(season_dir, f"{'-'.join(gad_name_parts)}.pkl")
    gad_state_filename_tmp = gad_state_filename + ".tmp"
    gad_state_ini_filename = os.path.join(season_dir, f"{'-'.join(gad_name_parts)}.ini")

    # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    num_generations = 100
    sol_per_pop = 500
    num_parents_mating = ceil(sol_per_pop / 2)
    keep_elitism = 5

    thread_pool = ThreadPoolExecutor(max_workers=processes)
    try:
        fitness_func, genes, gene_space, gene_type, tuning_progress, fitness_json_path = setup_gad(
            process_pool=process_pool, thread_pool=thread_pool, files=files, workdir=workdir, dry_run=dry_run,
            force=force, comskip_defaults=comskip_defaults,
            expensive_genes=expensive_genes, check_compute=check_compute, num_generations=num_generations,
            experimental=experimental,
            file_sample_size=15,
        )
    except UserWarning as e:
        logger.warning(e.args[0])
        thread_pool.shutdown(cancel_futures=True)
        return

    initial_solution_values = GENE_INITIAL_SOLUTION_VALUES.copy()

    # scan for other seasons to include comskip.ini as initial solutions
    for comskip_search_root, _, comskip_search_files in os.walk(os.path.dirname(season_dir)):
        for file in filter(lambda f: f == 'comskip.ini', comskip_search_files):
            comskip_ini_path = os.path.join(comskip_search_root, file)
            initial_solution_values_existing = initial_solution_values_from_ini(comskip_ini_path)
            if initial_solution_values_existing:
                logger.info(f"Initial solution values from {comskip_ini_path}: {initial_solution_values_existing}")
                initial_solution_values.append(initial_solution_values_existing)

    initial_solutions = [item for sublist in
                         map(lambda s: generate_initial_solutions(genes, s), initial_solution_values) for item in
                         sublist]

    for s in initial_solutions:
        logger.info("Initial solution : {solution}".format(solution=solution_repl(genes, s)))

    convergence_gauge = progress.gauge("CONV")
    convergence_gauge.renderer = lambda v: f"{v:.2f}"

    def gen_callback(ga_instance: pygad.GA):
        best_fitness = ga_instance.best_solutions_fitness
        if len(best_fitness) > 1:
            convergence_gauge.value(stdev(best_fitness))

        ga_instance_save(ga_instance, gad_state_filename_tmp)
        shutil.move(gad_state_filename_tmp, gad_state_filename)

        if ga_instance.best_solutions:
            write_ini_from_solution(gad_state_ini_filename, genes, ga_instance.best_solutions[-1], True)

        return None

    try:
        ga_in: dict[str, Any] = dict()
        with open(gad_state_filename, 'rb') as file:
            ga_in = cloudpickle.load(file)
        num_generations = max(1, int(ga_in.get('num_generations', num_generations)) - int(ga_in.get('generations_completed', 0)))
        initial_solutions = ga_in.get('population', initial_solutions)
        logging.info("Resuming from %s, %d generations left", gad_state_filename, num_generations)
    except FileNotFoundError:
        pass
    except BaseException as e:
        logging.debug("Error resuming from %s, starting over", gad_state_filename, e)
        logging.warning("Error resuming from %s, starting over", gad_state_filename)

    # ensure we have the desired population size
    additional_solutions_needed = max(0, sol_per_pop - len(initial_solutions))
    if additional_solutions_needed > 0:
        # Generate additional random solutions
        ga_temp = pygad.GA(num_generations=num_generations,
                           num_parents_mating=ceil(additional_solutions_needed / 2),
                           fitness_func=fitness_func,
                           sol_per_pop=additional_solutions_needed,
                           num_genes=len(genes),
                           gene_space=gene_space.copy(),
                           gene_type=gene_type.copy(),
                           suppress_warnings=True,
                           )
        additional_solutions = ga_temp.initial_population

        # Concatenate the initial and additional solutions
        initial_population = [numpy.array(sol, additional_solutions[0].dtype) for sol in initial_solutions] + list(
            additional_solutions)
    else:
        initial_population = initial_solutions

    # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           initial_population=initial_population,
                           num_genes=len(genes),
                           gene_space=gene_space,
                           gene_type=gene_type,
                           mutation_type="adaptive",
                           mutation_percent_genes=[25, 12],
                           parent_selection_type="sss",
                           keep_elitism=keep_elitism,
                           save_best_solutions=True,  # 2024-03-05 results are better with False
                           suppress_warnings=True,
                           on_generation=gen_callback,
                           )

    # restore state
    for attr_name in filter(lambda e: e in ga_in, GA_INSTANCE_ATTR_SAVE):
        attr_value = ga_in[attr_name]
        setattr(ga_instance, attr_name, attr_value)

    ga_instance.run()
    tuning_progress.stop()
    solution, solution_fitness, solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)

    thread_pool.shutdown()

    logger.info(
        "Parameters of the best solution : {solution}".format(solution=solution_repl(genes, solution)))
    logger.info("Fitness value of the best solution = {solution_fitness}".format(
        solution_fitness=solution_fitness))
    logger.debug("Best solutions = {best}\n   fitness = {fitness}".format(best=ga_instance.best_solutions,
                                                                          fitness=ga_instance.best_solutions_fitness))
    logger.info("Best fitness reached at generation %d", ga_instance.best_solution_generation)

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
            if isinstance(genes[idx].space, list) and len(genes[idx].space) == len(val):
                logger.info(f"{config} removing because all gene space is part of the solution space")
                solution[idx] = None
        else:
            logger.info(f"{config} DID contribute to fitness: {val}")

    logger.info(
        "Parameters of the adjusted solution : {solution}".format(solution=solution_repl(genes, solution)))

    write_ini_from_solution(target_comskip_ini, genes, solution,
                            comskip_defaults=comskip_defaults,
                            write_complete_config=True)

    if fitness_json_path and os.path.exists(fitness_json_path):
        shutil.move(fitness_json_path, os.path.join(season_dir, 'fitness.json'))

    if os.path.isfile(gad_state_filename):
        os.remove(gad_state_filename)
    if os.path.isfile(gad_state_ini_filename):
        os.remove(gad_state_ini_filename)

    return_code = common.ReturnCodeReducer()
    for filepath in files:
        process_pool.apply_async(common.pool_apply_wrapper(comchap), (filepath, filepath),
                                 {'force': True,
                                  'delete_edl': False,
                                  'backup_edl': True,
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

    time_start = time.time()
    time_progress = progress.progress("time limit", 0, time_limit)
    time_progress.renderer = common.s_to_ts

    return_code = common.ReturnCodeReducer()
    process_pool = Pool(processes=processes)

    def single_file_tune(f):
        if is_tuned(f, workdir) and force == 0:
            logger.info("%s already tuned, use --force to force tuning", f)
        elif not common.is_from_dvr(common.find_input_info(f)):
            logger.info("Skipping %s because it does not look like a DVR", f)
        else:
            process_pool.apply_async(common.pool_apply_wrapper(comtune), (f,),
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
            logger.info("INFO: not enough compute available")
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
                        process_pool.close()
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
                                tune_show(season_dir=season_dir, process_pool=process_pool, files=video_files,
                                          workdir=workdir,
                                          dry_run=dry_run, force=force, expensive_genes=expensive_genes,
                                          check_compute=check_compute, processes=processes, experimental=False)
                        except KeyboardInterrupt as e:
                            raise e
                        except BaseException as e:
                            logger.error(f"Skipping show {season_dir}, uncaught exception: {str(e)}", exc_info=e)
                    else:
                        for filepath in video_files:
                            if should_stop():
                                process_pool.close()
                                return return_code.code()
                            single_file_tune(filepath)
            if should_stop():
                process_pool.close()
                return return_code.code()

        process_pool.close()
    except KeyboardInterrupt:
        logger.info("User interrupt, waiting for pool to finish")
        return_code.set_code(130)
        process_pool.terminate()
    finally:
        common.pool_join_with_timeout(process_pool)
        time_progress.stop()

    return return_code.code()


if __name__ == '__main__':
    sys.exit(comtune_cli(sys.argv[1:]))
