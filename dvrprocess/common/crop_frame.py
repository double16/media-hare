import logging
import math
import re
import subprocess
from enum import Enum
from typing import Union

from . import get_video_width, get_video_height
from . import tools, constants

logger = logging.getLogger(__name__)


class CropFrameOperation(Enum):
    NONE = 0
    DETECT = 1
    NTSC = 2
    PAL = 3
    FIXED = 4


# https://en.wikipedia.org/wiki/List_of_common_resolutions
HD_RESOLUTIONS = [(1280, 720), (1280, 1080), (1440, 1080), (1920, 1080), (3840, 2160), (7680, 4320)]
CROP_FRAME_RESOLUTIONS = {
    CropFrameOperation.NTSC: [(640, 480), (704, 480), (720, 480)] + HD_RESOLUTIONS,
    CropFrameOperation.PAL: [(544, 576), (704, 576), (720, 576)] + HD_RESOLUTIONS
}


def find_crop_frame_filter(crop_frame_op: CropFrameOperation, input_info: dict, frame_rate,
                           crop_frame_fixed: Union[str, None] = None) -> Union[None, str]:
    """
    Find crop frame dimensions
    cropdetect output (ffmpeg v4, v5) looks like:
    [Parsed_cropdetect_1 @ 0x7f8ba9010d40] x1:246 x2:1676 y1:9 y2:1079 w:1424 h:1056 x:250 y:18 pts:51298 t:51.298000 crop=1424:1056:250:18
    :return: crop video filter or None
    """
    if crop_frame_op == CropFrameOperation.NONE:
        return None
    if crop_frame_op == CropFrameOperation.FIXED and crop_frame_fixed:
        if crop_frame_fixed.count(':') > 1:
            return crop_frame_fixed

    crop_frame_filter = None

    width = get_video_width(input_info)
    height = get_video_height(input_info)
    duration = float(input_info['format']['duration'])
    filename = input_info.get(constants.K_FORMAT, {}).get("filename")

    if crop_frame_op == CropFrameOperation.FIXED and crop_frame_fixed:
        crop_frame_fixed_parts = list(map(int, crop_frame_fixed.split(':')))
        final_width = min(width, crop_frame_fixed_parts[0])
        final_height = min(height, crop_frame_fixed_parts[1])
        if final_width == width and final_height == height:
            logger.info("crop frame filter is None")
            return None
        if len(crop_frame_fixed_parts) > 2:
            center_x = int(crop_frame_fixed_parts[2])
        else:
            center_x = math.floor((width - final_width) / 2)
        center_x = min(max(0, center_x), width - final_width)
        if len(crop_frame_fixed_parts) > 3:
            center_y = int(crop_frame_fixed_parts[3])
        else:
            center_y = math.floor((height - final_height) / 2)
        center_y = min(max(0, center_y), height - final_height)
        crop_frame_filter = f"crop={final_width}:{final_height}:{center_x}:{center_y}"
        logger.info("crop frame filter is %s", crop_frame_filter)
        return crop_frame_filter

    logger.info("Running crop frame detection")
    crop_frame_rect_histo = {}
    crop_detect_command = ['-hide_banner', '-skip_frame', 'nointra', '-i', filename,
                           '-vf', f'cropdetect=limit=0.15:reset={math.ceil(eval(frame_rate) * 3)}',
                           '-f', 'null', '/dev/null']
    crop_detect_process = tools.ffmpeg.Popen(crop_detect_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                             universal_newlines=True)
    crop_detect_regex = r't:([0-9.]+)\s+(crop=[0-9]+:[0-9]+:[0-9]+:[0-9]+)\b'
    crop_line_last_t = None
    crop_line_last_filter = None
    for line in crop_detect_process.stderr:
        m = re.search(crop_detect_regex, line)
        if m:
            crop_line_this_t = float(m.group(1))
            crop_line_this_filter = m.group(2)
            if crop_line_last_t is not None:
                if crop_line_this_filter == crop_line_last_filter:
                    crop_frame_rect_histo[crop_line_this_filter] = crop_frame_rect_histo.get(crop_line_this_filter,
                                                                                             0) + (
                                                                           crop_line_this_t - crop_line_last_t)
            crop_line_last_t = crop_line_this_t
            crop_line_last_filter = crop_line_this_filter

            # print(f"crop detect: {int(crop_line_this_t * 100 / duration)}%", end="\033[0G")

    # Filter out undesirable frames
    # 1. Only crop if the width is cropped a noticeable amount. Widescreen content intentionally has top and bottom black frames.
    # 2. Crop is over 10% the duration
    logger.debug("crop raw histo %s", crop_frame_rect_histo)
    crop_frame_rect_histo = dict(
        filter(lambda e: (width - get_crop_filter_parts(e[0])[0] >= 40) and e[1] > duration / 10,
               crop_frame_rect_histo.items()))
    logger.debug("crop filtered histo %s", crop_frame_rect_histo)
    if len(crop_frame_rect_histo) > 0:
        crop_frame_rect_histo_list = list(crop_frame_rect_histo.items())
        crop_frame_rect_histo_list.sort(key=lambda e: e[1], reverse=True)
        logger.debug("crop detect histo %s", crop_frame_rect_histo_list)
        crop_frame_filter = crop_frame_rect_histo_list[0][0]
        if width - get_crop_filter_parts(crop_frame_filter)[0] < 40:
            crop_frame_filter = None
    logger.info("crop frame detection found %s", crop_frame_filter)
    if crop_frame_op in CROP_FRAME_RESOLUTIONS:
        if crop_frame_filter is None:
            # assume cropping with centered rect
            rect = (width, height, 0, 0)
        else:
            rect = get_crop_filter_parts(crop_frame_filter)
        target_res = \
            sorted(CROP_FRAME_RESOLUTIONS[crop_frame_op], key=lambda e: abs(rect[0] - e[0]) + abs(rect[1] - e[1]))[
                0]
        logger.info("crop frame resolution match is %s", target_res)
        if target_res[0] == width and target_res[1] == height:
            crop_frame_filter = None
        else:
            crop_frame_filter = f"crop={target_res[0]}:{target_res[1]}:{max(0, int(((width - target_res[0]) / 2)))}:{max(0, int(((height - target_res[1]) / 2)))}"
    logger.info("crop frame filter is %s", crop_frame_filter)

    return crop_frame_filter


def get_crop_filter_parts(crop_filter):
    """
    Split the parts of the crop filter into integers.
    :param crop_filter: similar to crop=100:100:20:8
    :return: ints [width, height, x, y]
    """
    if crop_filter is None:
        return None
    return [int(i) for i in crop_filter.split('=')[1].split(':')]
