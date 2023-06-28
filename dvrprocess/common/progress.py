import logging
import time
from math import ceil
from typing import Union

_logger = logging.getLogger(__name__)


class Progress(object):

    def __init__(self, task: str):
        self.task: str = task
        self._start: int = 0
        self._end: int = 0
        """ Current progress by percentage (whole number), updated by progress(...), may be None """
        self.pct: Union[None, int] = None
        """ Updated by progress(...), may be None """
        self.eta: Union[None, float] = None
        """ Records when start(...) was called. """
        self._start_time: Union[None, float] = None
        """ The last time progress was reported. """
        self._last_progress_time: Union[None, float] = None
        """ The last percent we logged, used to keep the noise down. """
        self._last_pct: int = -1
        """ The last time we logged, used to keep the noise down. """
        self._last_report_time: Union[None, float] = None

    def update_reporting(self) -> bool:
        """
        Updates internal stats before reporting. Always call this and check the return
        to determine if reporting to the user is recommended. The intent is to keep noise
        down. Reporting too often may also be a performance hit.
        :return: True to report to user.
        """
        if self.pct is not None and self.pct != self._last_pct:
            if self._last_report_time is None or time.time() > self._last_report_time + 1 or self.pct == 100:
                self._last_pct = self.pct
                self._last_report_time = time.time()
                return True
        return False

    def human_duration(self, t: float) -> str:
        hour = int(t / 3600.0)
        minute = int(t / 60) % 60
        second = ceil(t % 60.0)
        if hour > 0:
            return f"{hour:02d}:{minute:02d}:{second:02d}"
        else:
            return f"{minute:02d}:{second:02d}"

    def remaining_human_duration(self) -> str:
        if self.eta is None:
            return "??:??"
        return self.human_duration(max(0.0, self.eta - time.time()))

    def start(self, start: int, end: int, msg: Union[None, str] = None) -> None:
        self._start = start
        self._end = end
        self._start_time = time.time()

    def stop(self, msg: Union[None, str] = None) -> None:
        pass

    def progress(self, position: int, msg: Union[None, str] = None, start: Union[None, int] = None,
                 end: Union[None, int] = None) -> None:
        self._last_progress_time = time.time()
        if start is not None:
            self._start = start
        if end is not None:
            self._end = end
        if self._end is None or self._start is None or (self._end - self._start <= 0):
            self.pct = None
            self.eta = None
        else:
            self.pct = ceil(100 * (position / (self._end - self._start)))
            if 10 <= self.pct <= 100:
                remaining_s = (time.time() - self._start_time) / self.pct * (100.0 - self.pct)
                self.eta = time.time() + remaining_s
            else:
                self.eta = None


class ProgressLog(Progress):

    def __init__(self, task: str):
        super().__init__(task)

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
        duration = self.human_duration(max(0.0, time.time() - self._start_time))
        if msg is None:
            msg = "stopped"
        _logger.info("%s: %s, elapsed %s", self.task, msg, duration)

    def progress(self, position: int, msg: Union[str, None] = None, start: Union[int, None] = None, end: Union[int, None] = None) -> None:
        super().progress(position, msg, start, end)
        if self.update_reporting():
            if msg:
                _logger.info("%s %s%% %s - %s", self.task, self.pct, self.remaining_human_duration(), msg)
            else:
                _logger.info("%s %s%% %s", self.task, self.pct, self.remaining_human_duration())


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
    """
    Create a progress object for reporting. This delegates to the progress reporter to create the
    configured type of reporting. The object will be created and start(...) will be called.
    :param task: the name of the task, should always be shown to the user
    :param start: the start of the range of progress
    :param end: the end of the range of progress
    :param msg: optional message to show to the user
    :return: Progress object
    """
    return _progress_reporter.start(task, start, end, msg)
