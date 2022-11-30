import re
import subprocess
import threading
from multiprocessing import Semaphore

from . import config
from .proc_invoker import SubprocessProcInvoker, MockProcInvoker

disk_semaphore = Semaphore(config.get_global_config_int('background_limits', 'disk_processes', fallback=10))


def _ffmpeg_version_parser(path):
    _maybe_version = float(
        re.search(r"version (\d+[.]\d+)", subprocess.check_output([path, '-version'], text=True))[1])
    if int(_maybe_version) not in [4, 5]:
        raise FileNotFoundError('ffmpeg version [4,5] not found')
    return _maybe_version


ffmpeg = SubprocessProcInvoker('ffmpeg', _ffmpeg_version_parser, version_target=['4.', '5.'], semaphore=disk_semaphore)


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


def mock_all():
    global ffmpeg, ffprobe, comskip, comskip_gui, mkvpropedit, ccextractor, subtitle_edit
    ffmpeg = MockProcInvoker('ffmpeg')
    ffprobe = MockProcInvoker('ffprobe')
    comskip = MockProcInvoker('comskip')
    comskip_gui = MockProcInvoker('comskip-gui')
    mkvpropedit = MockProcInvoker('mkvpropedit')
    ccextractor = MockProcInvoker('ccextractor')
    subtitle_edit = MockProcInvoker('subtitle-edit')


def mock_verify_all():
    ffmpeg.verify()
    ffprobe.verify()
    comskip.verify()
    comskip_gui.verify()
    mkvpropedit.verify()
    ccextractor.verify()
    subtitle_edit.verify()


class AudioLayout(object):
    def __init__(self, name, channels: list[str], voice_channels: list[str]):
        self.name = name
        self.channels = channels
        self.voice_channels = voice_channels

    def __str__(self):
        return f"{self.name} - {'+'.join(self.channels)}"

    def __repr__(self):
        return f"{self.name}, all({'+'.join(self.channels)}), voice({'+'.join(self.voice_channels)})"


_ffmpeg_audio_layouts: list[AudioLayout] = []
_ffmpeg_audio_layouts_lock = threading.Lock()


def get_audio_layouts(refresh=False) -> list[AudioLayout]:
    _ffmpeg_audio_layouts_lock.acquire()

    if refresh or len(_ffmpeg_audio_layouts) == 0:
        _ffmpeg_audio_layouts.clear()
        matches = re.findall(r"(\S+)\s+([A-Za-z]*\+[A-Za-z]*\+[A-Za-z]*.*)",
                             ffmpeg.check_output(['-hide_banner', '-layouts'], text=True))
        for m in matches:
            name = m[0]
            channels_str = m[1]
            channels = channels_str.split('+')
            voice_channels = list(filter(lambda e: e in ['FC', 'FR', 'FL'], channels))
            _ffmpeg_audio_layouts.append(AudioLayout(name=name, channels=channels, voice_channels=voice_channels))

    _ffmpeg_audio_layouts_lock.release()
    return _ffmpeg_audio_layouts
