import logging
from pathlib import Path

import pysrt
from ass_parser import read_ass, write_ass
from . import fatal, CODEC_SUBTITLE_ASS, CODEC_SUBTITLE_SRT, CODEC_SUBTITLE_SUBRIP

logger = logging.getLogger(__name__)


def subtitle_codec_from_filename(f, subtitle_codec_hint: [None, str]):
    if subtitle_codec_hint is None:
        f_s = str(f).lower()
        if '.ass' in f_s or '.ssa' in f_s:
            subtitle_codec_hint = CODEC_SUBTITLE_ASS
        elif '.srt' in f_s or '.sub' in f_s:
            subtitle_codec_hint = CODEC_SUBTITLE_SRT
        else:
            fatal(f"INFO: Unknown subtitle {f}")
    return subtitle_codec_hint


def read_subtitle_data(subtitle_codec, f):
    subtitle_codec = subtitle_codec_from_filename(f, subtitle_codec)
    if subtitle_codec == CODEC_SUBTITLE_ASS:
        return read_ass(Path(f))
    elif subtitle_codec in [CODEC_SUBTITLE_SRT, CODEC_SUBTITLE_SUBRIP]:
        return pysrt.open(f)
    else:
        raise f"INFO: Unknown subtitle codec {subtitle_codec}"


def write_subtitle_data(subtitle_codec, f, data) -> [None, str]:
    subtitle_codec = subtitle_codec_from_filename(f, subtitle_codec)
    if subtitle_codec == CODEC_SUBTITLE_ASS:
        if f is None:
            return write_ass(data)
        write_ass(data, Path(f))
    elif subtitle_codec in [CODEC_SUBTITLE_SRT, CODEC_SUBTITLE_SUBRIP]:
        if f is None:
            return "\n".join([str(x) for x in data])
        data.save(Path(f), 'utf-8')
    else:
        raise f"INFO: Unknown subtitle codec {subtitle_codec}"
    return None


def read_subtitle_text(subtitle_codec: [None, str], f):
    subtitle_codec = subtitle_codec_from_filename(f, subtitle_codec)
    lines = []
    if subtitle_codec == CODEC_SUBTITLE_ASS:
        ass_data = read_ass(Path(f))
        for event in list(ass_data.events):
            lines.append(event.text)
    elif subtitle_codec in [CODEC_SUBTITLE_SRT, CODEC_SUBTITLE_SUBRIP]:
        srt_data = pysrt.open(f)
        for event in list(srt_data):
            lines.append(event.text)
    else:
        fatal(f"INFO: Unknown subtitle codec {subtitle_codec}")
    return "\n".join(lines)


def cmp_subtitle_text(subtitle_codec: [None, str], f1, f2):
    return read_subtitle_text(subtitle_codec, f1) == read_subtitle_text(subtitle_codec, f2)


def subtitle_cut(subtitle_data, start_seconds: float, end_seconds: [None, float] = None):
    if hasattr(subtitle_data, 'events'):
        events = subtitle_data.events
    else:
        events = subtitle_data
    start_millis = int(start_seconds * 1000.0)
    if end_seconds is None:
        end_millis = None
        duration_millis = None
    else:
        end_millis = int(end_seconds * 1000.0)
        duration_millis = end_millis - start_millis
    idx = len(events) - 1
    while idx >= 0:
        event = events[idx]
        if hasattr(event.start, 'ordinal'):
            start = event.start.ordinal
            end = event.end.ordinal
        else:
            start = event.start
            end = event.end
        if end > start_millis:
            if end_millis is not None and start > end_millis:
                # adjust time span
                event.start -= duration_millis
                event.end -= duration_millis
            else:
                del events[idx]
        idx -= 1
    # fix SRT indicies
    for idx, event in enumerate(events):
        try:
            event.index = idx + 1
        except AttributeError:
            pass
