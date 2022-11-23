import re
import subprocess

from .proc_invoker import SubprocessProcInvoker


def _ffmpeg_version_parser(path):
    _maybe_version = float(
        re.search(r"version (\d+[.]\d+)", subprocess.check_output([path, '-version'], text=True))[1])
    if int(_maybe_version) not in [4, 5]:
        raise FileNotFoundError('ffmpeg version [4,5] not found')
    return _maybe_version


ffmpeg = SubprocessProcInvoker('ffmpeg', _ffmpeg_version_parser, version_target=['4.', '5.'])


def _ffprobe_version_parser(path):
    _maybe_version = float(
        re.search(r"version (\d+[.]\d+)", subprocess.check_output([path, '-version'], text=True))[1])
    if int(_maybe_version) not in [4, 5]:
        raise FileNotFoundError('ffprobe version [4,5] not found')
    return _maybe_version


ffprobe = SubprocessProcInvoker('ffprobe', _ffprobe_version_parser, ffmpeg.version_target)

ccextractor = SubprocessProcInvoker('ccextractor', lambda path: float(
    re.search(r"CCExtractor ([\d.]+)", subprocess.check_output([path, '--version'], text=True))[
        1]))

subtitle_edit = SubprocessProcInvoker('subtitle-edit', version_target=['3.6.8'])

comskip = SubprocessProcInvoker('comskip')

comskip_gui = SubprocessProcInvoker('comskip-gui', required=False)

mkvpropedit = SubprocessProcInvoker('mkvpropedit')

vainfo = SubprocessProcInvoker('vainfo', required=False)
vainfo.install = lambda: False

nvidia_smi = SubprocessProcInvoker('nvidia-smi', required=False)
nvidia_smi.install = lambda: False
