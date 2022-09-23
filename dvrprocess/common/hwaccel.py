import _thread
import os
import logging
import subprocess
import re
from enum import Enum
from shutil import which
from . import find_ffmpeg

_once_lock = _thread.allocate_lock()
logger = logging.getLogger(__name__)


class HWAccelMethod(Enum):
    NONE = 0
    VAAPI = 1
    NVENC = 2


hwaccel_method = None
vaapi_encoders = []
nvenc_encoders = []


def find_hwaccel_method() -> HWAccelMethod:
    global hwaccel_method
    if hwaccel_method is None:
        _once_lock.acquire()
        try:
            if hwaccel_method is None:
                hwaccel_method = _find_hwaccel_method()
                logger.debug("hwaccel method = %s", hwaccel_method)
        finally:
            _once_lock.release()
    return hwaccel_method


def _find_hwaccel_method() -> HWAccelMethod:
    global vaapi_encoders, nvenc_encoders
    for line in subprocess.check_output([find_ffmpeg(), "-hide_banner", "-encoders"], stderr=subprocess.STDOUT,
                                        text=True).splitlines():
        m = re.search(r'\b\w+_vaapi\b', line)
        if m:
            vaapi_encoders.append(m.group(0))
        m = re.search(r'\b\w+_nvenc\b', line)
        if m:
            nvenc_encoders.append(m.group(0))

    logger.debug("vaapi encoders = %s", vaapi_encoders)
    logger.debug("nvenc encoders = %s", nvenc_encoders)

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

    vainfo = which("vainfo")
    if not (vainfo and os.access(vainfo, os.X_OK)):
        return HWAccelMethod.NONE

    vaapi_profiles = []
    try:
        for line in subprocess.check_output([vainfo, "-a"], stderr=subprocess.STDOUT, text=True).splitlines():
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

    nvidia_smi = which("nvidia-smi")
    if not (nvidia_smi and os.access(nvidia_smi, os.X_OK)):
        return HWAccelMethod.NONE

    try:
        gpus = subprocess.check_output([nvidia_smi, "--list-gpus"], text=True)
        return HWAccelMethod.NVENC if 'GPU ' in gpus and len(nvidia_smi) > 0 else HWAccelMethod.NONE
    except subprocess.CalledProcessError:
        return HWAccelMethod.NONE


def has_hw_codec(codec: str):
    if codec == 'h265':
        codec = 'hevc'
    method = find_hwaccel_method()
    if method == HWAccelMethod.NVENC:
        return f"{codec}_nvenc" in nvenc_encoders
    elif method == HWAccelMethod.VAAPI:
        return f"{codec}_vaapi" in vaapi_encoders


def hwaccel_threads():
    """
    Some hardware acceleration has a thread limit.
    :return:
    """
    # nvdec has a limit of decode surfaces, but decoding isn't CPU heavy, so let it fail to get maximum encode
    # if find_hwaccel_method() == HWAccelMethod.NVENC:
    # return ['-threads', str(min(8, core_count()))]
    return []


def hwaccel_decoding(codec: str):
    method = find_hwaccel_method()
    if method == HWAccelMethod.NVENC:
        return ["-hwaccel:v", "nvdec", "-hwaccel_device", "0"]
    elif method == HWAccelMethod.VAAPI:
        return ["-hwaccel:v", "vaapi", "-init_hw_device", "vaapi=vaapi:", "-hwaccel_device", "vaapi"]
    return []


def hwaccel_encoding(output_stream: str, codec: str, output_type: str, tune: str, preset: str, crf: int, qp: int,
                     target_bitrate: int):
    method = find_hwaccel_method()
    if method == HWAccelMethod.NVENC:
        return _nvenc_encoding(output_stream=output_stream, codec=codec, output_type=output_type, tune=tune,
                               preset=preset, crf=crf, qp=qp, target_bitrate=target_bitrate)
    elif method == HWAccelMethod.VAAPI:
        return _vaapi_encoding(output_stream=output_stream, codec=codec, output_type=output_type, tune=tune,
                               preset=preset, crf=crf, qp=qp, target_bitrate=target_bitrate)
    return []


def _nvenc_encoding(output_stream: str, codec: str, output_type: str, tune: str, preset: str, crf: int, qp: int,
                    target_bitrate: int):
    options = [f"-c:{output_stream}"]
    if codec in ['h265']:
        options.extend([f"hevc_nvenc"])
    else:
        options.extend([f"{codec}_nvenc"])

    if codec in ['h264', 'h265', 'hevc']:
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
            f'-qmax:{output_stream}', str(qp),
            f'-b:{output_stream}', '0'])

        options.extend([f'-multipass:{output_stream}', 'fullres'])
        if output_type != 'ts':
            options.extend([f'-a53cc:{output_stream}', 'false'])

    if codec in ['h264']:
        options.extend((f"-profile:{output_stream}", "high"))
    elif codec in ['h265', 'hevc']:
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
            f'-qmax:{output_stream}', str(qp),
            f'-b:{output_stream}', f'{target_bitrate}k'])

    if codec in ['h264']:
        options.extend([f"-profile:{output_stream}", "high"])
        options.extend([f"-quality:{output_stream}", "0"])

    return options
