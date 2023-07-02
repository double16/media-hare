import logging
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union

import pysrt
from ass_parser import read_ass, write_ass, AssFile, AssEvent
from pysrt import SubRipItem, SubRipFile, SubRipTime

from . import fatal, constants

logger = logging.getLogger(__name__)


def subtitle_codec_from_filename(f, subtitle_codec_hint: [None, str]):
    if subtitle_codec_hint is None:
        f_s = str(f).lower()
        if '.ass' in f_s or '.ssa' in f_s:
            subtitle_codec_hint = constants.CODEC_SUBTITLE_ASS
        elif '.srt' in f_s or '.sub' in f_s:
            subtitle_codec_hint = constants.CODEC_SUBTITLE_SRT
        else:
            fatal(f"INFO: Unknown subtitle {f}")
    return subtitle_codec_hint


def read_subtitle_data(subtitle_codec, f):
    subtitle_codec = subtitle_codec_from_filename(f, subtitle_codec)
    if subtitle_codec == constants.CODEC_SUBTITLE_ASS:
        return read_ass(Path(f))
    elif subtitle_codec in [constants.CODEC_SUBTITLE_SRT, constants.CODEC_SUBTITLE_SUBRIP]:
        return pysrt.open(f)
    else:
        raise f"INFO: Unknown subtitle codec {subtitle_codec}"


def write_subtitle_data(subtitle_codec, f, data) -> [None, str]:
    subtitle_codec = subtitle_codec_from_filename(f, subtitle_codec)
    if subtitle_codec == constants.CODEC_SUBTITLE_ASS:
        if f is None:
            return write_ass(data)
        write_ass(data, Path(f))
    elif subtitle_codec in [constants.CODEC_SUBTITLE_SRT, constants.CODEC_SUBTITLE_SUBRIP]:
        if f is None:
            return "\n".join([str(x) for x in data])
        data.save(Path(f), 'utf-8')
    else:
        raise f"INFO: Unknown subtitle codec {subtitle_codec}"
    return None


def read_subtitle_text(subtitle_codec: [None, str], f):
    subtitle_codec = subtitle_codec_from_filename(f, subtitle_codec)
    lines = []
    if subtitle_codec == constants.CODEC_SUBTITLE_ASS:
        ass_data = read_ass(Path(f))
        for event in list(ass_data.events):
            lines.append(event.text)
    elif subtitle_codec in [constants.CODEC_SUBTITLE_SRT, constants.CODEC_SUBTITLE_SUBRIP]:
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


class SubtitleElementFacade(ABC):
    def __init__(self):
        pass

    def __repr__(self):
        return f"({self.start(), self.end()} \"{self.text()}\""

    @abstractmethod
    def text(self) -> Union[str, None]:
        return None

    @abstractmethod
    def set_text(self, new_value: str):
        pass

    @abstractmethod
    def start(self) -> Union[int, None]:
        return None

    @abstractmethod
    def set_start(self, new_value: int):
        pass

    @abstractmethod
    def end(self) -> Union[int, None]:
        return None

    @abstractmethod
    def set_end(self, new_value: int):
        pass

    def duration(self) -> int:
        return max(self.end() - self.start(), 0)

    def index(self) -> int:
        """
        Return index of event in the subtitle file.
        """
        return 0

    def set_index(self, new_index: int):
        """
        Set the index of event in the subtitle file.
        """
        pass

    def move(self, new_start: int):
        d = self.duration()
        self.set_start(new_start)
        self.set_end(new_start + d)


class AssElementFacade(SubtitleElementFacade):
    def __init__(self, event: AssEvent):
        super().__init__()
        self.event = event

    def text(self) -> Union[str, None]:
        return self.event.text

    def set_text(self, new_value: str):
        self.event.text = new_value.replace('\n', '\\N')

    def start(self) -> Union[int, None]:
        return self.event.start

    def set_start(self, new_value: int):
        self.event.start = new_value

    def end(self) -> Union[int, None]:
        return self.event.end

    def set_end(self, new_value: int):
        self.event.end = new_value


class SubripElementFacade(SubtitleElementFacade):
    def __init__(self, event: SubRipItem):
        super().__init__()
        self.event = event

    def text(self) -> Union[str, None]:
        return self.event.text

    def set_text(self, new_value: str):
        self.event.text = new_value

    def start(self) -> Union[int, None]:
        return self.event.start.ordinal

    def set_start(self, new_value: int):
        self.event.start = SubRipTime.from_ordinal(new_value)

    def end(self) -> Union[int, None]:
        return self.event.end.ordinal

    def set_end(self, new_value: int):
        self.event.end = SubRipTime.from_ordinal(new_value)

    def index(self) -> int:
        return self.event.index

    def set_index(self, new_index: int):
        self.event.index = new_index


class SubtitleFileFacade(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def events(self):
        """
        Return a generator of SubtitleElementFacade
        :return:
        """
        yield None

    @abstractmethod
    def insert(self, index: int) -> SubtitleElementFacade:
        """
        Create a new event if the same type.
        """
        raise NotImplementedError()

    @abstractmethod
    def write(self, file: Path):
        raise NotImplementedError()


class AssFileFacade(SubtitleFileFacade):

    def __init__(self, file: AssFile):
        super().__init__()
        self.file = file

    def events(self):
        for event_idx, event in enumerate(self.file.events):
            yield event_idx, AssElementFacade(event)

    def insert(self, index: int) -> SubtitleElementFacade:
        event = AssEvent()
        self.file.events.insert(index, event)
        return AssElementFacade(event)

    def write(self, file: Path):
        write_ass(self.file, file)


class SubripFileFacade(SubtitleFileFacade):

    def __init__(self, file: SubRipFile):
        super().__init__()
        self.file = file

    def events(self):
        for event_idx, event in enumerate(self.file):
            yield event_idx, SubripElementFacade(event)

    def insert(self, index: int) -> SubtitleElementFacade:
        event = SubRipItem()
        self.file.insert(index, event)
        return SubripElementFacade(event)

    def write(self, file: Path):
        self.file.save(file, "utf-8")


def new_subtitle_file_facade(subtitle: Union[AssFile, SubRipFile]) -> SubtitleFileFacade:
    if isinstance(subtitle, AssFile):
        return AssFileFacade(subtitle)
    elif isinstance(subtitle, SubRipFile):
        return SubripFileFacade(subtitle)
    raise NotImplementedError(subtitle.__class__)


def open_subtitle_file_facade(file: Path) -> SubtitleFileFacade:
    if file.match("*.srt"):
        return new_subtitle_file_facade(pysrt.open(file))
    elif file.match(".ass") or file.match("*.ssa"):
        return new_subtitle_file_facade(read_ass(file))
    else:
        raise NotImplementedError("Unsupported subtitle %s" % file)
