import logging
import re
import subprocess
import threading
from multiprocessing import Semaphore

from . import config
from .proc_invoker import SubprocessProcInvoker, MockProcInvoker

logger = logging.getLogger(__name__)

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
    global ffmpeg, ffprobe, comskip, comskip_gui, mkvpropedit, ccextractor, subtitle_edit, _ffmpeg_audio_layouts
    ffmpeg = MockProcInvoker('ffmpeg')
    ffprobe = MockProcInvoker('ffprobe')
    comskip = MockProcInvoker('comskip')
    comskip_gui = MockProcInvoker('comskip-gui')
    mkvpropedit = MockProcInvoker('mkvpropedit')
    ccextractor = MockProcInvoker('ccextractor')
    subtitle_edit = MockProcInvoker('subtitle-edit')
    _ffmpeg_audio_layouts = []


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

    def map_to(self, target) -> dict[str, list[str]]:
        """
        Map channels to the target audio layout. The mapping may contain multiple channels.
        :param target:
        :return: key is target channel, value is zero or more source channels (zero indicates to drop the channel)
        """
        result = dict()
        # Handle equal channels
        target_remaining = []
        source_remaining = self.channels.copy()
        for target_channel in target.channels:
            if target_channel in self.channels:
                result[target_channel] = [target_channel]
                source_remaining.remove(target_channel)
            else:
                result[target_channel] = []
                target_remaining.append(target_channel)
        if 'LFE' in target_remaining:
            target_remaining.remove('LFE')  # nothing we can do for LFE
        # Handle channels based on Center, Left or Right. This won't work if there's multiples left
        for target_channel in target_remaining:
            position_ch = None
            for c in ['C', 'R', 'L']:
                if c in target_channel:
                    position_ch = c
            found = False
            if position_ch is not None:
                for source_channel in source_remaining:
                    if position_ch in source_channel:
                        result[target_channel] = [source_channel]
                        # target_remaining.remove(target_channel)
                        source_remaining.remove(source_channel)
                        found = True
                        break
            if not found:
                if 'C' in target_channel:
                    result[target_channel] = list(filter(lambda e: 'R' in e or 'L' in e, source_remaining))
                else:
                    result[target_channel] = list(filter(lambda e: 'C' in e, source_remaining))

        if len(source_remaining) > 0:
            logger.warning("Unmapped source channels: %s, mapping from %s, to %s", ",".join(source_remaining),
                           self.name, target.name)

        return result


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


def get_audio_layout_by_name(name: str) -> [None, AudioLayout]:
    for audio_layout in get_audio_layouts():
        if audio_layout.name == name:
            return audio_layout
    return None
