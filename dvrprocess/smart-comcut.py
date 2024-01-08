#!/usr/bin/env python3

import getopt
import logging
import math
import os
import sys
from statistics import stdev, mean
from typing import Union

import common
from comchap import comchap, get_expected_adjusted_duration
from comcut import comcut
from common import crop_frame, config, edl_util, progress

VIDEO_MIN_COUNT = 5

logger = logging.getLogger(__name__)


def usage():
    print(f"""{sys.argv[0]}

Cuts commercials only when a season fits closely within the average of length post-cut

-n, --dry-run
    Output command that would be used but do nothing
--verbose
-k, --keep
    Keep original file in a backup prefixed by ".~"
-c, --commercial-details
    Consider commercial durations for allowing cut. This seem to remove a lot of opportunity needlessly.
--strict
    Lower tolerances to improve accuracy but may miss cut opportunities
--sigma
    Sigma limit in seconds
-a, --all
    Cut all files regardless of fit
--work-dir={config.get_work_dir()}
--preset=veryslow,slow,medium,fast,veryfast
    Set ffmpeg preset, defaults to {config.get_global_config_option('ffmpeg', 'preset')}
--force-encode
    Force encoding to be precise, i.e. skip cutting by key frames
--crop-frame
    Detect and crop surrounding frame. Does not modify widescreen formats that have top and bottom frames.
--crop-frame-ntsc
    Detect and crop surrounding frame to one of the NTSC (and HD) common resolutions.
--crop-frame-pal
    Detect and crop surrounding frame to one of the PAL (and HD) common resolutions.
--crop-frame-fixed=w:h[:x:y]
    Crop frame to specified values. If x:y are omitted, the frame is centered.
-v, --vcodec=h264[,hevc,...]
    The video codec: {config.get_global_config_option('video', 'codecs')} (default), h265, mpeg2.
""", file=sys.stderr)


class CommercialRange(object):
    def __init__(self, values: list[float], error_range: list[float]):
        average = mean(values)
        self.values = values
        self.begin = average - error_range[0]
        self.end = average + error_range[1]
        self.average = average
        self.deviation = stdev(values)
        self.min_value = min(values)
        self.max_value = max(values)

    def __str__(self):
        return f"({common.s_to_ts(self.begin)}, {common.s_to_ts(self.end)}) σ{self.deviation} min/max({common.s_to_ts(self.min_value)} {common.s_to_ts(self.max_value)})"


class CommercialBreakStats(object):
    def __init__(self, start: CommercialRange, end: CommercialRange, duration: CommercialRange):
        self.start = start
        self.end = end
        self.duration = duration

    def __str__(self):
        return f"({self.start}, {self.end}) duration {self.duration}"


def smart_comcut_cli(argv):
    dry_run = False
    no_curses = False
    commercial_details = False
    strict = False
    all_videos = False
    keep = False
    workdir = config.get_work_dir()
    sigma = None
    preset = None
    force_encode = False
    crop_frame_op = crop_frame.CropFrameOperation.NONE
    crop_frame_fixed = None
    desired_video_codecs = None

    dvrconfig = list(
        filter(lambda e: "crop-frame" in e or "preset" in e, common.get_arguments_from_config(argv, '.dvrconfig')))

    try:
        opts, args = getopt.getopt(dvrconfig + list(argv), "nkcav:",
                                   ["help", "dry-run", "keep", "commercial-details", "strict", "all", "work-dir=",
                                    "verbose", "sigma=", "preset=", "force-encode", "crop-frame", "crop-frame-ntsc",
                                    "crop-frame-pal", "crop-frame-fixed=", "vcodec=", "no-curses"])
    except getopt.GetoptError:
        usage()
        return 255
    for opt, arg in opts:
        if opt == '--help':
            usage()
            return 255
        elif opt in ("-n", "--dry-run"):
            dry_run = True
            no_curses = True
        elif opt == "--no-curses":
            no_curses = True
        elif opt in ("-k", "--keep"):
            keep = True
        elif opt in ("-c", "--commercial-details"):
            commercial_details = True
        elif opt == "--strict":
            strict = True
        elif opt in ("-a", "--all"):
            all_videos = True
        elif opt == "--work-dir":
            workdir = arg
        elif opt == "--sigma":
            sigma = float(arg)
        elif opt == "--verbose":
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt == "--force-encode":
            force_encode = True
        elif opt in ("-p", "--preset"):
            preset = arg
        elif opt == "--crop-frame":
            crop_frame_op = crop_frame.CropFrameOperation.DETECT
        elif opt == "--crop-frame-ntsc":
            crop_frame_op = crop_frame.CropFrameOperation.NTSC
        elif opt == "--crop-frame-pal":
            crop_frame_op = crop_frame.CropFrameOperation.PAL
        elif opt == "--crop-frame-fixed":
            crop_frame_op = crop_frame.CropFrameOperation.FIXED
            crop_frame_fixed = arg
        elif opt in ("-v", "--vcodec"):
            desired_video_codecs = arg.split(',')

    if len(args) < 1:
        usage()
        return 255

    if not workdir:
        workdir = config.get_work_dir()

    common.cli_wrapper(smart_comcut_cli_run, args=args, dry_run=dry_run, keep=keep, workdir=workdir, preset=preset,
                       force_encode=force_encode, crop_frame_op=crop_frame_op, crop_frame_fixed=crop_frame_fixed,
                       desired_video_codecs=desired_video_codecs,
                       commercial_details=commercial_details, strict=strict, all_videos=all_videos, sigma=sigma,
                       no_curses=no_curses)


def smart_comcut_cli_run(args: list, dry_run, keep, workdir, preset, force_encode,
                         crop_frame_op, crop_frame_fixed, desired_video_codecs, commercial_details, strict, all_videos, sigma) -> int:
    for arg in args:
        for root, dirs, files in os.walk(arg, topdown=True):
            dirs.sort()
            videos = []
            show_label = None
            for file in files:
                filepath = os.path.abspath(os.path.join(root, file))
                filename = os.path.basename(filepath)
                if filename.startswith('.') or not filename.endswith(".mkv"):
                    continue
                video_info = common.find_input_info(filepath, raise_errors=False)
                if not video_info:
                    continue
                show_label = os.path.sep.join(os.path.abspath(filepath).split(os.path.sep)[-3:-1])
                episode_count, duration, total_duration = common.episode_info(video_info)
                adjusted_duration = total_duration

                # Missing EDL can indicate no commercials, we want to include the duration in our averages
                edl_path = common.edl_for_video(filepath)
                has_com = False
                commercial_break_count = 0
                commercial_breaks = []
                if common.is_from_dvr(video_info):
                    if not os.access(edl_path, os.R_OK):
                        cc_return_code = comchap(filepath, filepath, delete_edl=False, modify_video=False)
                        if cc_return_code == 255:
                            return cc_return_code
                    if os.access(edl_path, os.R_OK):
                        cuts = edl_util.parse_edl_cuts(edl_path)
                        for idx, event in enumerate(cuts):
                            has_com = True
                            this_duration = (event.end - event.start)
                            if this_duration >= 20 or (event.start > 0 and idx < len(cuts) - 1):
                                commercial_break_count += 1
                                commercial_breaks.append(event)
                            adjusted_duration -= this_duration
                # remove the following else to consider already cut durations
                else:
                    continue

                adjusted_duration /= episode_count

                videos.append({
                    'filepath': filepath,
                    'info': video_info,
                    'duration': duration,
                    'adjusted_duration': adjusted_duration,
                    'has_com': has_com,
                    'commercial_break_count': commercial_break_count,
                    'commercial_breaks': commercial_breaks,
                })
                logger.debug(f"{filepath}: {common.seconds_to_timespec(duration)}"
                             f", {common.seconds_to_timespec(adjusted_duration)}"
                             f", {commercial_break_count} commercials")

            if len(videos) == 0:
                continue

            # Ignore special longer episodes, such as the pilot
            episode_common_duration = common.get_common_episode_duration(list(map(lambda e: e['info'], videos)))
            expected_adjusted_duration = get_expected_adjusted_duration(videos[0]['info'])
            logger.info("%s: target episode duration %s, expected episode duration %s", show_label,
                        common.seconds_to_timespec(episode_common_duration),
                        common.seconds_to_timespec(expected_adjusted_duration))
            videos = list(filter(lambda e: episode_common_duration == common.round_episode_duration(e['info']), videos))

            if len(videos) == 0:
                continue
            if len(videos) < VIDEO_MIN_COUNT:
                logger.error(f"{show_label}: Need at least {VIDEO_MIN_COUNT} videos to compute average")
                continue

            # Sort and ignore top and bottom durations numbers for average. Probably should use sigma instead.
            videos.sort(key=lambda v: v['duration'])
            video_count_to_trim = int(math.ceil(len(videos) / 10.0))
            # videos_for_stats = videos[video_count_to_trim:-video_count_to_trim]
            videos_for_stats = videos

            durations = list(map(lambda v: v['duration'], videos_for_stats))
            adjusted_durations = list(map(lambda v: v['adjusted_duration'], videos_for_stats))
            average_duration = mean(durations)
            stdev_duration = stdev(durations)
            average_adjusted_duration = mean(adjusted_durations)
            stdev_adjusted_duration = stdev(adjusted_durations)

            error_range = duration_error_range_seconds(average_duration, strict, sigma)
            accepted_range = [average_adjusted_duration - error_range[0], average_adjusted_duration + error_range[1]]

            logger.info(f"{show_label}: "
                        f"count = {len(adjusted_durations)}, "
                        f"average = {common.seconds_to_timespec(average_duration)} σ{stdev_duration}, "
                        f"average adjusted = {common.seconds_to_timespec(average_adjusted_duration)} "
                        f"σ{common.seconds_to_timespec(stdev_adjusted_duration)}")

            # TODO: organize breaks into "slots" with similar start times. If there is a break starting at 0, use the
            # length to adjust start time

            # Compute the expected number of commercial breaks
            commercial_break_histo = {}
            commercial_break_histo_data_count = 0
            # We only consider commercial counts for videos that need cut
            for video in filter(lambda v: v['has_com'] and v['duration'] > accepted_range[1], videos_for_stats):
                commercial_break_histo_data_count += 1
                key = video['commercial_break_count']
                if key in commercial_break_histo:
                    commercial_break_histo[key] += 1
                else:
                    commercial_break_histo[key] = 1
            logger.info(f"{show_label}: commercial_break_histo = {commercial_break_histo}")
            commercial_break_count_min = math.floor((commercial_break_histo_data_count * 9) / 10)
            target_commercial_breaks = dict(
                filter(lambda x: x[1] >= commercial_break_count_min, commercial_break_histo.items()))
            if len(target_commercial_breaks) != 1:
                logger.log(logging.ERROR if commercial_details else logging.WARNING,
                    f"{show_label}: Cannot find consistent commercial break counts, min histo count = {commercial_break_count_min}")
                if commercial_details:
                    continue
            else:
                required_commercial_break_count = list(target_commercial_breaks.items())[0][0]
                logger.info(f"{show_label}: required_commercial_break_count = {required_commercial_break_count}")

                # build acceptable commercial break lengths
                required_commercial_breaks = []
                videos_for_commercials = list(
                    filter(lambda v: v['commercial_break_count'] == required_commercial_break_count, videos_for_stats))
                for i in range(required_commercial_break_count):
                    start_array = []
                    end_array = []
                    duration_array = []
                    for video in videos_for_commercials:
                        break_i = video['commercial_breaks'][i]
                        start_array.append(break_i.start)
                        end_array.append(break_i.end)
                        duration_array.append(break_i.end - break_i.start)
                    start_range = CommercialRange(start_array, error_range)
                    end_range = CommercialRange(end_array, error_range)
                    duration_range = CommercialRange(duration_array, [20.0, 20.0])
                    required_commercial_breaks.append(CommercialBreakStats(start_range, end_range, duration_range))

                required_commercial_breaks_str = ""
                for c in required_commercial_breaks:
                    required_commercial_breaks_str += f"\n{c}"
                logger.debug("%s: required_commercial_breaks = %s", show_label, required_commercial_breaks_str)

            if not all_videos:
                abort = False
                if (average_adjusted_duration - stdev_adjusted_duration < accepted_range[
                    0] or average_adjusted_duration + stdev_adjusted_duration > accepted_range[1]):
                    logger.error("%s: σ%s outside of accepted range", show_label,
                                   common.seconds_to_timespec(stdev_adjusted_duration))
                    abort = True
                expected_adjusted_duration_diff = abs(average_adjusted_duration - expected_adjusted_duration)
                if expected_adjusted_duration_diff > (error_range[1] * 2):
                    logger.error("%s: %s adjusted duration outside of accepted range", show_label,
                                   common.seconds_to_timespec(expected_adjusted_duration_diff))
                    abort = True
                if abort:
                    continue

            videos.sort(key=lambda v: v['filepath'])
            videos_with_com = list(filter(lambda v: v['has_com'], videos))
            videos_progress = progress.progress("files", 0, len(videos_with_com))
            for video_idx, video in enumerate(videos_with_com):
                adjusted_duration = video['adjusted_duration']
                filepath = video['filepath']
                this_commercial_break_count = video['commercial_break_count']
                duration_ok = accepted_range[0] < adjusted_duration < accepted_range[1]
                if commercial_details:
                    # TODO: account for optional break at beginning or ending, depends on when recording started
                    # TODO: adjust start time based on break starting at 0
                    if this_commercial_break_count == required_commercial_break_count:
                        commercials_ok = True
                        commercial_breaks = video['commercial_breaks']
                        for i, val in enumerate(commercial_breaks):
                            expected = required_commercial_breaks[i]
                            if not expected.start.begin < val.start < expected.start.end:
                                logger.debug(f"{filepath}: break {i} start {val.start} out of range {expected.start}")
                                commercials_ok = False
                            if not expected.end.begin < val.end < expected.end.end:
                                logger.debug(f"{filepath}: break {i} end {val.end} out of range {expected.end}")
                                commercials_ok = False
                            if not expected.duration.begin < (val.end - val.start) < expected.duration.end:
                                logger.debug(
                                    f"{filepath}: break {i} duration {val.end - val.start} out of range {expected.duration}")
                                commercials_ok = False
                    else:
                        commercials_ok = False
                else:
                    commercials_ok = None
                if all_videos or (duration_ok and commercials_ok in [None, True]):
                    logger.debug(
                        f"{filepath} will be cut, {common.seconds_to_timespec(adjusted_duration - average_adjusted_duration)}")
                    if dry_run:
                        logger.info("cut %s", filepath)
                    else:
                        cut(filepath, keep=keep, workdir=workdir, preset=preset, force_encode=force_encode,
                            crop_frame_op=crop_frame_op, crop_frame_fixed=crop_frame_fixed,
                            desired_video_codecs=desired_video_codecs)
                else:
                    logger.error(
                        f"{filepath} comskip FAILURE, {common.seconds_to_timespec(adjusted_duration)}"
                        f", {common.seconds_to_timespec(adjusted_duration - average_adjusted_duration)}"
                        f", {this_commercial_break_count} commercials")
                videos_progress.progress(video_idx)
            videos_progress.stop()

    return 0


def duration_error_range_seconds(average_duration, strict=False, sigma=None):
    if sigma is not None:
        return [sigma, sigma]

    # It seems reasonable a short show should have less deviation. That isn't the case in practice. Some of the
    # deviation is from the DVR start/stop time, which is based on minute accuracy. Average commercial length time
    # doesn't deviate much based on show length, but rather the number of commercials.
    #
    # if average_duration < 1850:
    #     if strict:
    #         return [30, 30]
    #     else:
    #         return [60, 60]

    if strict:
        return [60, 60]
    else:
        return [120, 120]


def cut(filepath, keep=False, workdir=None, preset=None, force_encode=False,
        crop_frame_op: crop_frame.CropFrameOperation = crop_frame.CropFrameOperation.NONE,
        crop_frame_fixed: Union[str, None] = None,
        desired_video_codecs=None):
    filepath_stat = os.stat(filepath)
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    tempfilename = f"{dirname}/.~{basename.replace('.mkv', '.transcoded.mkv')}"
    if common.assert_not_transcoding(filepath, exit=False) != 0:
        return
    cut_return_code = comcut(filepath, tempfilename, delete_edl=False, force_clear_edl=True, workdir=workdir,
                             preset=preset, force_encode=force_encode, crop_frame_op=crop_frame_op,
                             crop_frame_fixed=crop_frame_fixed,
                             desired_video_codecs=desired_video_codecs)
    if cut_return_code != 0:
        if os.path.exists(tempfilename):
            os.remove(tempfilename)
        return
    # if os.stat(tempfilename).st_size < (filepath_stat.st_size / 2):
    #     logger.error(f"Cut file is less than half of original, skipping")
    #     os.remove(tempfilename)
    #     return

    if keep:
        keeppath = f"{dirname}/.~{basename}"
        try:
            os.replace(filepath, keeppath)
        except OSError:
            logger.error(f"Failed to keep file: {filepath} to {keeppath}")
            return

    try:
        os.replace(tempfilename, filepath)
    except OSError:
        logger.error(f"Failed to move converted file: {tempfilename} to {filepath}")

    return


if __name__ == '__main__':
    os.nice(12)
    sys.exit(smart_comcut_cli(sys.argv[1:]))
