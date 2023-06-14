import logging
from math import ceil
from typing import Union

_logger = logging.getLogger(__name__)


class Progress(object):

    def __init__(self, task: str):
        self.task = task
        self._start = 0
        self._end = 0
        self.pct = None

    def start(self, start: int, end: int, msg: Union[None, str] = None) -> None:
        self._start = start
        self._end = end

    def stop(self, msg: Union[None, str] = None) -> None:
        pass

    def progress(self, position: int, msg: Union[None, str] = None, start: Union[None, int] = None,
                 end: Union[None, int] = None) -> None:
        if start is not None:
            self._start = start
        if end is not None:
            self._end = end
        if self._end is None or self._start is None or (self._end - self._start <= 0):
            self.pct = None
        else:
            self.pct = ceil(100 * (position / (self._end - self._start)))


class ProgressLog(Progress):

    def __init__(self, task: str):
        super().__init__(task)
        self._last_pct = -1

    def start(self, start: int, end: int, msg: Union[str, None] = None) -> None:
        super().start(start, end, msg)
        if end > start:
            self.progress(start, msg, start, end)
        elif msg is not None:
            _logger.info("%s: %s", self.task, msg)
        else:
            _logger.info("%s: starting", self.task)

    def stop(self, msg: Union[str, None] = None) -> None:
        super().stop(msg)
        if msg is not None:
            _logger.info("%s: %s", self.task, msg)
        else:
            _logger.info("%s: stopped", self.task)

    def progress(self, position: int, msg: Union[str, None] = None, start: Union[int, None] = None, end: Union[int, None] = None) -> None:
        super().progress(position, msg, start, end)
        if self.pct is not None and self.pct != self._last_pct:
            self._last_pct = self.pct
            _logger.info("%s %s %%", self.task, self.pct)


class ProgressReporter(object):
    def __init__(self):
        pass

    def _create_progress(self, task: str) -> Progress:
        return ProgressLog(task)

    def start(self, task: str, start: int, end: int, msg: Union[None, str] = None) -> Progress:
        p = self._create_progress(task)
        p.start(start, end, msg)
        return p


_progress_reporter = ProgressReporter()


def progress(task: str, start: int, end: int, msg: Union[None, str] = None) -> Progress:
    return _progress_reporter.start(task, start, end, msg)
