import _thread
import logging
import os
import re
import subprocess
import sys
from shutil import which

_allocate_lock = _thread.allocate_lock
_once_lock = _allocate_lock()

logger = logging.getLogger(__name__)


def _fatal(message):
    logger.fatal(message)
    sys.exit(255)


ffmpeg_path = None
ffmpeg_version: float = 0.0


def find_ffmpeg():
    global ffmpeg_path, ffmpeg_version
    if ffmpeg_path is None:
        _once_lock.acquire()
        try:
            if ffmpeg_path is None:
                _maybe_path = _find_ffmpeg()
                _maybe_version = float(
                    re.search(r"version (\d+[.]\d+)", subprocess.check_output([_maybe_path, '-version'], text=True))[1])
                if int(_maybe_version) not in [4, 5]:
                    raise FileNotFoundError('ffmpeg version [4,5] not found')
                ffmpeg_path = _maybe_path
                ffmpeg_version = _maybe_version
        finally:
            _once_lock.release()
    return ffmpeg_path


def _find_ffmpeg():
    # Never use Plex's version. We have no guarantees on version or patches applied. Keep the code so people know we tried.
    ffmpeg = "x /usr/lib/plexmediaserver/Plex\\ Transcoder"

    if os.access(ffmpeg, os.X_OK):
        ld_library_path = "/usr/lib/plexmediaserver:/usr/lib/plexmediaserver/lib"
    else:
        ffmpeg = which("ffmpeg")
    if len(ffmpeg) == 0:
        _fatal("'ffmpeg' not found")
    if not os.access(ffmpeg, os.X_OK):
        _fatal(f"{ffmpeg} is not an executable")

    return ffmpeg


ffprobe_path = None
ffprobe_version: float = 0.0


def find_ffprobe():
    global ffprobe_path, ffprobe_version
    if ffprobe_path is None:
        _once_lock.acquire()
        try:
            if ffprobe_path is None:
                _maybe_path = _find_ffprobe()
                _maybe_version = float(
                    re.search(r"version (\d+[.]\d+)", subprocess.check_output([_maybe_path, '-version'], text=True))[1])
                if int(_maybe_version) not in [4, 5]:
                    raise FileNotFoundError('ffprobe version [4,5] not found')
                ffprobe_path = _maybe_path
                ffprobe_version = _maybe_version
        finally:
            _once_lock.release()
    return ffprobe_path


def _find_ffprobe():
    ffprobe = which("ffprobe")
    if not os.access(ffprobe, os.X_OK):
        _fatal(f"{ffprobe} is not an executable")
    return ffprobe


ccextractor_path = None
ccextractor_version: float = 0.0


def find_ccextractor():
    global ccextractor_path, ccextractor_version
    if ccextractor_path is None:
        _once_lock.acquire()
        try:
            if ccextractor_path is None:
                _maybe_path = _find_ccextractor()
                _maybe_version = float(
                    re.search(r"CCExtractor ([\d.]+)", subprocess.check_output([_maybe_path, '--version'], text=True))[
                        1])
                ccextractor_path = _maybe_path
                ccextractor_version = _maybe_version
        finally:
            _once_lock.release()
    return ccextractor_path


def _find_ccextractor():
    ccextractor = which("ccextractor")
    if not os.access(ccextractor, os.X_OK):
        _fatal(f"{ccextractor} is not an executable")
    return ccextractor
