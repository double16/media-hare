import logging
import multiprocessing
import threading
import time
from math import ceil
from multiprocessing import Queue
from typing import Union, Dict

_logger = logging.getLogger(__name__)


class Progress(object):

    def __init__(self, task: str):
        self.task: str = task
        self._start: int = 0
        self._end: int = 0
        self.last_position: int = 0
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
        """ Function to render progress position as string """
        self.renderer = None

    def get_last_progress_time(self) -> Union[None, float]:
        return self._last_report_time

    def update_reporting(self) -> bool:
        """
        Updates internal stats before reporting. Always call this and check the return
        to determine if reporting to the user is recommended. The intent is to keep noise
        down. Reporting too often may also be a performance hit.
        :return: True to report to user.
        """
        if (self.pct is not None and self.pct != self._last_pct) or self.renderer is not None:
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

    def elapsed_human_duration(self) -> str:
        return self.human_duration(max(0.0, time.time() - self._start_time))

    def start(self, start: int, end: int, msg: Union[None, str] = None) -> None:
        self._start = start
        self._end = end
        self._start_time = time.time()

    def stop(self, msg: Union[None, str] = None) -> None:
        pass

    def progress(self, position: int, msg: Union[None, str] = None, start: Union[None, int] = None,
                 end: Union[None, int] = None) -> None:
        self._last_progress_time = time.time()
        self.last_position = position
        if start is not None:
            self._start = start
        if end is not None:
            self._end = end
        if self._end is None or self._start is None or (self._end - self._start <= 0):
            self.pct = None
            self.eta = None
        else:
            self.pct = ceil(100 * (position / (self._end - self._start)))
            if self.pct <= 100 and (self.pct >= 10 or (time.time() - self._start_time) > 60):
                remaining_s = (time.time() - self._start_time) / self.pct * (100.0 - self.pct)
                # TODO: save time for each 10% and use weighed calculation
                self.eta = time.time() + remaining_s
            else:
                self.eta = None

    def position_str(self, position: int) -> str:
        if self.renderer is None:
            if self.pct is not None and 0 < self.pct <= 100:
                return f"{self.pct:>3}%"
            return ""
        return self.renderer(position)


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
        if msg is None:
            msg = "stopped"
        _logger.info("%s: %s, elapsed %s", self.task, msg, self.elapsed_human_duration())

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


def set_progress_reporter(new_reporter: ProgressReporter):
    global _progress_reporter
    _progress_reporter = new_reporter


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


_PROGRESS_BY_TASK: Dict[str, Progress] = dict()


class ProgressStartMessage(object):
    """
    Object intended to be placed on the queue.
    """
    def __init__(self, task: str, start: int, end: int, msg: Union[None, str] = None):
        self.task = task
        self.start = start
        self.end = end
        self.msg = msg

    def apply(self):
        _PROGRESS_BY_TASK[self.task] = progress(self.task, self.start, self.end, self.msg)


class ProgressProgressMessage(object):
    """
    Object intended to be placed on the queue.
    """
    def __init__(self, task: str, position: int, msg: Union[None, str] = None, start: Union[None, int] = None,
                 end: Union[None, int] = None):
        self.task = task
        self.position = position
        self.msg = msg
        self.start = start
        self.end = end

    def apply(self):
        try:
            _PROGRESS_BY_TASK.get(self.task).progress(self.position, self.msg, self.start, self.end)
        except KeyError:
            pass


class ProgressStopMessage(object):
    """
    Object intended to be placed on the queue.
    """
    def __init__(self, task: str, msg: Union[str, None] = None):
        self.task = task
        self.msg = msg

    def apply(self):
        try:
            _PROGRESS_BY_TASK.get(self.task).stop(self.msg)
            _PROGRESS_BY_TASK.pop(self.task)
        except KeyError:
            pass


class LogMessage(object):
    def __init__(self, record: logging.LogRecord):
        self.record = record

    def apply(self):
        logging.root.callHandlers(self.record)


class SubprocessLogHandler(logging.Handler):
    def __init__(self, queue: Queue):
        super().__init__(logging.DEBUG)
        self.queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        self.queue.put_nowait(LogMessage(record))


class SubprocessProgress(Progress):
    """
    Sends messages to the queue.
    """
    def __init__(self, task: str, queue: Queue):
        super().__init__(task)
        self.queue = queue

    def start(self, start: int, end: int, msg: Union[None, str] = None) -> None:
        self.queue.put_nowait(ProgressStartMessage(self.task, start, end, msg))

    def progress(self, position: int, msg: Union[None, str] = None, start: Union[None, int] = None,
                 end: Union[None, int] = None) -> None:
        self.queue.put_nowait(ProgressProgressMessage(self.task, position, msg, start, end))

    def stop(self, msg: Union[None, str] = None) -> None:
        self.queue.put_nowait(ProgressStopMessage(self.task, msg))


class SubprocessProgressReporter(ProgressReporter):

    def __init__(self, progress_queue: Queue):
        super().__init__()
        self.queue = progress_queue

    def _create_progress(self, task: str) -> Progress:
        return SubprocessProgress(task, self.queue)


_PROGRESS_QUEUE = None


def _progress_queue_feed(q: Queue):
    """
    Processes progress events from the queue.
    """
    while True:
        try:
            m = q.get(True)
        except ValueError:
            # queue is closed
            return
        try:
            m.apply()
        except Exception as e:
            _logger.error("processing progress", e)


def setup_parent_progress() -> Queue:
    """
    Setup this process to receive progress from subprocesses.
    """
    global _PROGRESS_QUEUE
    if _PROGRESS_QUEUE is not None:
        return _PROGRESS_QUEUE
    _PROGRESS_QUEUE = multiprocessing.Manager().Queue()
    thread = threading.Thread(target=_progress_queue_feed, args=(_PROGRESS_QUEUE,), name="Progress Processor")
    thread.daemon = True
    thread.start()
    return _PROGRESS_QUEUE


_subprocess_progress_configured = False


def setup_subprocess_progress(progress_queue: Queue):
    """
    Setup this process to send progress to the parent process.
    """
    global _subprocess_progress_configured
    if _subprocess_progress_configured:
        return
    set_progress_reporter(SubprocessProgressReporter(progress_queue))
    logging.root.addHandler(SubprocessLogHandler(progress_queue))
    logging.root.setLevel(logging.DEBUG)
    _subprocess_progress_configured = True
