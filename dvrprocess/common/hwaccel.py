"""
Hardware acceleration support.
Use as follows:

hwaccel_configure("none"|"full")
cmd = []
cmd.extend(hwaccel_threads())
cmd.extend(hwaccel_prologue('mpeg2video', 'h264'))
cmd.extend(hwaccel_decoding('mpeg2video'))
cmd.extend(hwaccel_encoding(output_stream='1', codec='h264', output_type='mkv', tune=None, preset='veryslow', crf=23, qp=28,
                     target_bitrate=1200))

"""
import _thread
import json
import logging
import re
import subprocess
from enum import Enum
from typing import Union

import numpy

from . import tools

_once_lock = _thread.allocate_lock()
logger = logging.getLogger(__name__)


class HWAccelMethod(Enum):
    NONE = 0
    VAAPI = 1
    NVENC = 2
    VIDEO_TOOLBOX = 3


class HWAccelRequest(Enum):
    """
    The user requested hardware acceleration mode.
    """
    NONE = 0  # Disable hardware acceleration
    AUTO = 1  # Automatically choose, but tend towards low use of hardware
    FULL = 2  # Use full decode and encode acceleration
    VAAPI = 3  # Use VAAPI decode and encode acceleration, if available, otherwise software
    NVENC = 4  # Use nvidia decode and encode acceleration, if available, otherwise software
    VIDEO_TOOLBOX = 5  # Use macos decode and encode acceleration, if available, otherwise software


hwaccel_method = None
hwaccel_requested = HWAccelRequest.AUTO
vaapi_encoders = []
nvenc_encoders = []
video_toolbox_encoders = []


def hwaccel_configure(user_option: Union[None, str], forgiving: bool = False) -> HWAccelRequest:
    """
    Convert the hwaccel option as a string to an enum. Defaults to AUTO. Sets the use of acceleration in the
    other methods.
    :param user_option: hwaccel config option
    :param forgiving: True for best effort to forgive errors
    """
    global hwaccel_requested
    hwaccel_requested = _hwaccel_configure(user_option, forgiving=forgiving)
    logger.info("hwaccel configured as %s", hwaccel_requested)
    return hwaccel_requested


def _hwaccel_configure(user_option: Union[None, str], forgiving: bool = False) -> HWAccelRequest:
    """
    Convert the hwaccel option as a string to an enum. Defaults to AUTO.
    :param user_option: hwaccel config option
    :param forgiving: True for best effort to forgive errors
    """
    # if forgiving:
    #     NVENC works well with damaged frames
    #     return HWAccelRequest.NONE

    if user_option is None:
        return HWAccelRequest.AUTO
    user_option = user_option.upper()
    if user_option in ['NONE', 'FALSE', 'OFF']:
        return HWAccelRequest.NONE
    for option in HWAccelRequest:
        if user_option == option.name:
            return option
    return HWAccelRequest.AUTO


def find_hwaccel_method() -> HWAccelMethod:
    global hwaccel_method
    if hwaccel_method is None:
        _once_lock.acquire()
        try:
            if hwaccel_method is None:
                hwaccel_method = _find_hwaccel_method()
                logger.info("hwaccel method = %s", hwaccel_method)
        finally:
            _once_lock.release()
    return hwaccel_method


def _find_hwaccel_method() -> HWAccelMethod:
    global vaapi_encoders, nvenc_encoders, video_toolbox_encoders
    for line in tools.ffmpeg.check_output(["-hide_banner", "-encoders"], stderr=subprocess.STDOUT,
                                          text=True).splitlines():
        m = re.search(r'\b\w+_vaapi\b', line)
        if m:
            vaapi_encoders.append(m.group(0))
        m = re.search(r'\b\w+_nvenc\b', line)
        if m:
            nvenc_encoders.append(m.group(0))
        m = re.search(r'\b\w+_videotoolbox\b', line)
        if m:
            video_toolbox_encoders.append(m.group(0))

    logger.debug("vaapi encoders = %s", vaapi_encoders)
    logger.debug("nvenc encoders = %s", nvenc_encoders)
    logger.debug("videotoolbox encoders = %s", video_toolbox_encoders)

    if len(video_toolbox_encoders) > 0:
        return HWAccelMethod.VIDEO_TOOLBOX

    result = _find_nvenc_method()
    if result != HWAccelMethod.NONE:
        return result

    return _find_vaapi_method()


def _find_vaapi_method() -> HWAccelMethod:
    """
    Find vaapi encoding. Encoder specific options can be found with `ffmpeg -h encoder=h264_vaapi`.
    :return:
    """

    global vaapi_encoders

    if not tools.vainfo.present():
        return HWAccelMethod.NONE

    vaapi_profiles = []
    try:
        for line in tools.vainfo.check_output(["-a"], stderr=subprocess.STDOUT, text=True).splitlines():
            m = re.search(r'(?:\b|^)(VAProfile\w+(?:Main|High))(?:\b|/)', line)
            if m:
                vaapi_profiles.append(m.group(1))
    except subprocess.CalledProcessError:
        return HWAccelMethod.NONE

    logger.debug("vaapi profiles = %s", vaapi_profiles)

    for encoder in list(vaapi_encoders):
        encoder_short = re.sub(r'_vaapi$', '', encoder)
        encoder_short = re.sub(r'video$', '', encoder_short).upper()
        if f"VAProfile{encoder_short}Main" not in vaapi_profiles and f"VAProfile{encoder_short}High" not in vaapi_profiles:
            vaapi_encoders.remove(encoder)

    return HWAccelMethod.VAAPI if len(vaapi_encoders) > 0 else HWAccelMethod.NONE


def _find_nvenc_method() -> HWAccelMethod:
    """
    Find nvidia encoding. Encoder specific options can be found with `ffmpeg -h encoder=h264_nvenc`.
    :return:
    """

    global nvenc_encoders

    if not tools.nvidia_smi.present():
        return HWAccelMethod.NONE

    try:
        gpus = tools.nvidia_smi.check_output(["--list-gpus"], text=True)
        return HWAccelMethod.NVENC if 'GPU ' in gpus else HWAccelMethod.NONE
    except subprocess.CalledProcessError:
        return HWAccelMethod.NONE


def has_hw_codec(codec: str) -> bool:
    """
    Check if the current hardware acceleration setting has an accelerated version of the codec.
    """
    if codec == 'h265':
        codec = 'hevc'
    method = find_hwaccel_method()
    if method == HWAccelMethod.NVENC:
        return f"{codec}_nvenc" in nvenc_encoders
    elif method == HWAccelMethod.VAAPI:
        return f"{codec}_vaapi" in vaapi_encoders
    elif method == HWAccelMethod.VIDEO_TOOLBOX:
        return f"{codec}_videotoolbox" in video_toolbox_encoders
    return False


def require_hw_codec(codec: str) -> bool:
    """
    Check if the codec requires use of hardware encoding.
    :param codec:
    :return:
    """
    return codec in ['h265', 'hevc']


def ffmpeg_sw_codec(desired_codec: str) -> str:
    """
    Get the ffmpeg software codec.
    """
    if re.search("h[0-9][0-9][0-9]", desired_codec):
        return f"libx{desired_codec[1:]}"
    if desired_codec == "opus":
        return "libopus"
    return desired_codec


def hwaccel_threads() -> list[str]:
    """
    Some hardware acceleration has a thread limit.
    :return:
    """
    # nvdec has a limit of decode surfaces, but decoding isn't CPU heavy, so let it fail to get maximum encode
    # if find_hwaccel_method() == HWAccelMethod.NVENC:
    # return ['-threads', str(min(8, core_count()))]
    return []


def hwaccel_prologue(input_video_codec: str, target_video_codec: [None, str]) -> list[str]:
    """
    Return ffmpeg options for initializing hwaccel.
    :param input_video_codec:
    :param target_video_codec: target codec OR None OR "copy" if encoding isn't necessary
    :return:
    """
    global hwaccel_requested

    if hwaccel_requested == HWAccelRequest.NONE:
        return []

    method = find_hwaccel_method()
    result = []

    if hwaccel_requested == HWAccelRequest.AUTO and target_video_codec is not None and target_video_codec != 'copy' and require_hw_codec(
            target_video_codec):
        logger.info("Switching hwaccel to full for video codec %s", target_video_codec)
        hwaccel_requested = HWAccelRequest.FULL

    if hwaccel_requested == HWAccelRequest.AUTO:
        # let ffmpeg figure it out
        result.extend(['-hwaccel', 'auto'])
    elif hwaccel_requested in [HWAccelRequest.FULL, HWAccelRequest.NVENC] and method == HWAccelMethod.NVENC:
        result.extend(
            ["-hwaccel:v", "nvdec", "-init_hw_device", "cuda=cuda", "-hwaccel_device", "0", "-filter_hw_device",
             "cuda"])
    elif hwaccel_requested in [HWAccelRequest.FULL, HWAccelRequest.VAAPI] and method == HWAccelMethod.VAAPI:
        result.extend(["-hwaccel:v", "vaapi", "-init_hw_device", "vaapi=vaapi:", "-hwaccel_device", "vaapi"])

    return result


def hwaccel_decoding(codec: str) -> list[str]:
    """
    Return ffmpeg options for using hwaccel for decoding.
    """
    # Nothing specific at this time, ffmpeg seems to choose a hardware decoder automatically
    return []


def hwaccel_required_hwupload_filter() -> bool:
    """
    Determine if the 'hwupload' video filter is required.
    :return:
    """
    global hwaccel_requested
    if hwaccel_requested == HWAccelRequest.NONE:
        return False
    method = find_hwaccel_method()
    return method not in [HWAccelMethod.NONE, HWAccelMethod.VIDEO_TOOLBOX]


def hwaccel_encoding(output_stream: str, codec: str, output_type: str, tune: [None, str], preset: str, crf: int,
                     qp: int,
                     target_bitrate: int) -> (list[str], HWAccelMethod):
    """
    Return ffmpeg options for using hwaccel for encoding.
    :param output_stream: ffmpeg output stream spec, '1', ...
    :param codec: 'h264', 'h265', ...
    :param output_type: 'mkv', 'mp4', ...
    :param tune: ffmpeg tune value: None, 'animation', etc.
    :param preset: 'veryslow', 'slow', etc.
    :param crf: 23, 31, ...
    :param qp: 28, 34, ...
    :param target_bitrate: in kbps, such as 1200, 3500, ...
    """
    if preset == "copy":
        return [f"-c:{output_stream}", "copy"], HWAccelMethod.NONE

    method = find_hwaccel_method()
    if hwaccel_requested in [HWAccelRequest.FULL,
                             HWAccelRequest.NVENC] and method == HWAccelMethod.NVENC and has_hw_codec(codec):
        return (_nvenc_encoding(output_stream=output_stream, codec=codec, output_type=output_type, tune=tune,
                                preset=preset, crf=crf, qp=qp, target_bitrate=target_bitrate), HWAccelMethod.NVENC)
    elif hwaccel_requested in [HWAccelRequest.FULL,
                               HWAccelRequest.VAAPI] and method == HWAccelMethod.VAAPI and has_hw_codec(codec):
        return (_vaapi_encoding(output_stream=output_stream, codec=codec, output_type=output_type, tune=tune,
                                preset=preset, crf=crf, qp=qp, target_bitrate=target_bitrate), HWAccelMethod.VAAPI)
    elif hwaccel_requested in [HWAccelRequest.FULL,
                               HWAccelRequest.VIDEO_TOOLBOX] and method == HWAccelMethod.VIDEO_TOOLBOX and has_hw_codec(
        codec):
        return (_video_toolbox_encoding(output_stream=output_stream, codec=codec, output_type=output_type, tune=tune,
                                        preset=preset, crf=crf, qp=qp, target_bitrate=target_bitrate),
                HWAccelMethod.VIDEO_TOOLBOX)
    else:
        return (_sw_encoding(output_stream=output_stream, codec=codec, output_type=output_type, tune=tune,
                             preset=preset, crf=crf, qp=qp, target_bitrate=target_bitrate), HWAccelMethod.NONE)


def _nvenc_encoding(output_stream: str, codec: str, output_type: str, tune: str, preset: str, crf: int, qp: int,
                    target_bitrate: int):
    # https://docs.nvidia.com/video-technologies/video-codec-sdk/nvenc-video-encoder-api-prog-guide/

    options = [f"-c:{output_stream}"]
    if codec in ['h265']:
        options.extend([f"hevc_nvenc"])
    else:
        options.extend([f"{codec}_nvenc"])

    if codec in ['h264', 'h265', 'hevc']:
        options.extend([f'-tune:{output_stream}', 'hq'])

        preset_opt = f"-preset:{output_stream}"
        if preset in ['veryslow', 'slowest']:
            options.extend([preset_opt, 'p7'])
        elif preset in ['slower']:
            options.extend([preset_opt, 'p6'])
        elif preset in ['slow']:
            options.extend([preset_opt, 'p5'])
        elif preset in ['medium']:
            options.extend([preset_opt, 'p4'])
        elif preset in ['fast']:
            options.extend([preset_opt, 'p3'])
        elif preset in ['faster']:
            options.extend([preset_opt, 'p2'])
        elif preset in ['veryfast', 'fastest']:
            options.extend([preset_opt, 'p1'])

        # options.extend([f'-rc:{output_stream}', 'constqp', '-qp', str(qp)])
        # ... OR ...
        # nvenc equivalent of -crf (?), https://github.com/HandBrake/HandBrake/issues/2231
        options.extend([
            f'-rc:{output_stream}', 'vbr',
            f'-cq:{output_stream}', str(qp),
            f'-qmin:{output_stream}', str(qp),
            f'-qmax:{output_stream}', str(qp + 6),
            f'-b:{output_stream}', '0'])

        options.extend([f'-multipass:{output_stream}', 'fullres'])
        # https://docs.nvidia.com/video-technologies/video-codec-sdk/nvenc-video-encoder-api-prog-guide/#look-ahead
        options.extend([f'-rc-lookahead:{output_stream}', '32'])
        # https://docs.nvidia.com/video-technologies/video-codec-sdk/nvenc-video-encoder-api-prog-guide/#b-frames-as-reference
        options.extend([f'-b_ref_mode:{output_stream}', '2'])
        # https://docs.nvidia.com/video-technologies/video-codec-sdk/nvenc-video-encoder-api-prog-guide/#adaptive-quantization-aq
        options.extend([f'-spatial-aq:{output_stream}', '1'])
        options.extend([f'-temporal-aq:{output_stream}', '1'])
        options.extend([f'-aq-strength:{output_stream}', '15'])

        if output_type != 'ts':
            options.extend([f'-a53cc:{output_stream}', 'false'])

    if codec in ['h264']:
        options.extend((f"-profile:{output_stream}", "high"))
    elif codec in ['h265', 'hevc']:
        # TODO: if 10-bit depth, use 'main-10'
        options.extend((f"-profile:{output_stream}", "main"))

    return options


def _vaapi_encoding(output_stream: str, codec: str, output_type: str, tune: str, preset: str, crf: int, qp: int,
                    target_bitrate: int):
    options = [f"-c:{output_stream}"]
    if codec in ['h265']:
        options.extend(["hevc_vaapi"])
    else:
        options.extend([f"{codec}_vaapi"])

    if codec in ['h264', 'h265', 'hevc']:
        options.extend([f"-rc_mode:{output_stream}", "VBR",
                        f"-qp:{output_stream}", str(qp),
                        f'-qmin:{output_stream}', str(qp),
                        f'-qmax:{output_stream}', str(qp + 2),
                        f'-b:{output_stream}', f'{target_bitrate}k'])

    if codec in ['h264']:
        options.extend([f"-profile:{output_stream}", "high"])
        options.extend([f"-quality:{output_stream}", "0"])

    return options


def _video_toolbox_encoding(output_stream: str, codec: str, output_type: str, tune: str, preset: str, crf: int, qp: int,
                            target_bitrate: int):
    options = [f"-c:{output_stream}"]
    if codec in ['h265']:
        options.extend(["hevc_videotoolbox"])
    else:
        options.extend([f"{codec}_videotoolbox"])

    # It appears the encoder defaults to optimize for size (?) but quality isn't great
    if codec in ['h264', 'h265', 'hevc']:
        options.extend([f'-prio_speed:{output_stream}', 'false'])
        options.extend([f'-b:{output_stream}', f'{target_bitrate}k'])

    if codec in ['h264']:
        options.extend([f"-profile:{output_stream}", "high"])
        if output_type != 'ts':
            options.extend([f'-a53cc:{output_stream}', 'false'])

    if codec == 'h265':
        # https://trac.ffmpeg.org/wiki/Encode/H.265
        options.extend([f'-tag:{output_stream}', 'hvc1'])

    return options


def _sw_encoding(output_stream: str, codec: str, output_type: str, tune: str, preset: str, crf: int, qp: int,
                 target_bitrate: int):
    options = [f"-c:{output_stream}", ffmpeg_sw_codec(codec),
               f"-crf:{output_stream}", str(crf),
               f"-preset:{output_stream}", preset]
    if codec == 'h264' and tune is not None:
        options.extend([f"-tune:{output_stream}", tune])
    # Do not copy Closed Captions, they will be extracted into a subtitle stream
    if codec == 'h264' and output_type != 'ts':
        options.extend([f"-a53cc:{output_stream}", '0'])
    if codec == 'h265':
        # https://trac.ffmpeg.org/wiki/Encode/H.265
        options.extend([f'-tag:{output_stream}', 'hvc1'])
    return options


def hwaccel_gpustat() -> tuple[Union[float, None], Union[float, None]]:
    """
    Get the GPU compute % and memory % usage
    :return: (compute 0-100, memory 0-100)
    """
    global hwaccel_requested
    if hwaccel_requested == HWAccelRequest.NVENC:
        return _nvenc_gpustat()
    return None, None


def _nvenc_gpustat() -> tuple[Union[float, None], Union[float, None]]:
    if not tools.nvidia_gpustat.present():
        return None, None

    try:
        stat = json.loads(tools.nvidia_gpustat.check_output(["--json"], text=True))
        compute = []
        memory = []
        for gpu in stat["gpus"]:
            compute.append(float(gpu["utilization.gpu"]))
            memory.append(float(gpu["memory.used"])*100.0/float(gpu["memory.total"]))
        if len(compute) == 0:
            # no gpus
            return None, None
        return numpy.average(compute), numpy.average(memory)
    except Exception as e:
        logger.warning("Could not parse gpustat result", e)
        pass

    return None, None
