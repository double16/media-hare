import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from math import ceil
from multiprocessing import Queue
from typing import Union, Dict

import psutil

from . import hwaccel

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
        """ The last time we reported, used to keep the noise down. """
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
        if position is None:
            return
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
            if self.pct <= 100 and (self.pct >= 10 or (self.pct >= 2 and (time.time() - self._start_time) > 60)):
                remaining_s = (time.time() - self._start_time) / self.pct * (100.0 - self.pct)
                # TODO: save time for each 10% and use weighed calculation
                self.eta = time.time() + remaining_s
            else:
                self.eta = None

    def progress_inc(self, value: int = 1, msg: Union[None, str] = None, start_inc: Union[None, int] = None,
                     end_inc: Union[None, int] = None) -> None:
        if value > 0:
            start = self._start
            if start_inc is not None:
                start += start_inc
            end = self._end
            if end_inc is not None:
                end += end_inc
            self.progress(self.last_position + value, msg=msg, start=start, end=end)

    def end_inc(self, value: int = 1, msg: Union[None, str] = None):
        self.progress(self.last_position, msg=msg, end=self._end + value)

    def position_str(self, position: int) -> str:
        if self.renderer is None:
            if self.pct is not None and 0 < self.pct <= 100:
                return f"{self.pct:>3}%"
            return ""
        return self.renderer(position)

    def range(self) -> tuple[int, int]:
        return self._start, self._end


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
            msg = "complete"
        _logger.info("%s: %s, elapsed %s", self.task, msg, self.elapsed_human_duration())

    def progress(self, position: int, msg: Union[str, None] = None, start: Union[int, None] = None, end: Union[int, None] = None) -> None:
        super().progress(position, msg, start, end)
        if self.update_reporting():
            if self.renderer is None:
                rendered = str(self.pct) + "%"
            else:
                rendered = self.renderer(position)
            if msg:
                _logger.info("%s %s %s - %s", self.task, rendered, self.remaining_human_duration(), msg)
            else:
                _logger.info("%s %s %s", self.task, rendered, self.remaining_human_duration())


class Gauge(object):
    """
    A gauge is a point in time float measurement.
    """

    def __init__(self, name: str, low: float = None, high: float = None, renderer=None):
        self.name = name
        self.low: float = low
        self.high: float = high
        self.normal_range: tuple[float, float] = None
        self.warning_range: tuple[float, float] = None
        self.critical_range: tuple[float, float] = None
        self.last_value: Union[None, float] = None
        """ The last time a value was received. """
        self._last_value_time: Union[None, float] = None
        """ The last time we reported, used to keep the noise down. """
        self._last_report_time: Union[None, float] = None
        """ Function to render the value as string """
        self.renderer = renderer

    def value(self, value: float):
        self.last_value = value
        self._last_value_time = time.time()

    def update_reporting(self) -> bool:
        """
        Always call this and check the return to determine if reporting to the user is recommended. The
        intent is to keep noise down. Reporting too often may also be a performance hit.
        :return: True to report to user.
        """
        if self.last_value is not None:
            if self._last_report_time is None or time.time() > self._last_report_time + 1:
                self._last_report_time = time.time()
                return True
        return False

    def value_str(self, value: float) -> str:
        if value is None:
            return ""
        if self.renderer is None:
            return str(value)
        return self.renderer(value)

    def is_normal(self, value: float = None) -> bool:
        if value is None:
            value = self.last_value
        if value is None:
            return False
        if self.normal_range is None:
            return True
        return self.normal_range[0] <= value <= self.normal_range[1]

    def is_warning(self, value: float) -> bool:
        if value is None:
            value = self.last_value
        if value is None:
            return False
        if self.warning_range is None:
            return False
        return self.warning_range[0] <= value <= self.warning_range[1]

    def is_critical(self, value: float) -> bool:
        if value is None:
            value = self.last_value
        if value is None:
            return False
        if self.critical_range is None:
            return False
        return self.critical_range[0] <= value <= self.critical_range[1]


class GaugeLog(Gauge):
    """
    Gauge that logs the value.
    """

    def __init__(self, name: str, low: float = None, high: float = None):
        super().__init__(name, low, high)

    def value(self, value: float):
        super().value(value)
        if self.update_reporting():
            level = logging.INFO
            if self.is_critical(value):
                level = logging.CRITICAL
            elif self.is_warning(value):
                level = logging.WARNING
            _logger.log(level, "%s %s", self.name, self.value_str(value))


class ProgressReporter(object):
    def __init__(self):
        pass

    def _create_progress(self, task: str) -> Progress:
        return ProgressLog(task)

    def start(self, task: str, start: int, end: int, msg: Union[None, str] = None) -> Progress:
        p = self._create_progress(task)
        p.start(start, end, msg)
        return p

    def _create_gauge(self, name: str, low: float = None, high: float = None) -> Gauge:
        return GaugeLog(name, low, high)

    def gauge(self, name: str, low: float = None, high: float = None) -> Gauge:
        return self._create_gauge(name, low, high)


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


def gauge(name: str, low: float = None, high: float = None) -> Gauge:
    """
    Create a gauge object for reporting. This delegates to the progress reporter to create the
    configured type of reporting.
    :return: Gauge object
    """
    return _progress_reporter.gauge(name, low, high)


_PROGRESS_BY_TASK: Dict[str, Progress] = dict()
_GAUGE_BY_NAME: Dict[str, Gauge] = dict()


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
            _PROGRESS_BY_TASK[self.task].progress(self.position, self.msg, self.start, self.end)
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
            p = _PROGRESS_BY_TASK.pop(self.task)
            if p:
                p.stop(self.msg)
        except KeyError:
            pass


class GaugeCreateMessage(object):
    """
    Object intended to be placed on the queue.
    """

    def __init__(self, name: str, low: float, high: float):
        self.name = name
        self.low = low
        self.high = high

    def apply(self):
        try:
            _GAUGE_BY_NAME[self.name] = gauge(self.name, self.low, self.high)
        except KeyError:
            pass


class GaugeValueMessage(object):
    """
    Object intended to be placed on the queue.
    """

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def apply(self):
        try:
            _GAUGE_BY_NAME[self.name].value(self.value)
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


class SubprocessGauge(Gauge):
    def __init__(self, name: str, low: float, high: float, queue: Queue):
        super().__init__(name, low, high)
        self.queue = queue
        self.queue.put_nowait(GaugeCreateMessage(name, low, high))

    def value(self, value: float):
        self.queue.put_nowait(GaugeValueMessage(self.name, value))


class SubprocessProgressReporter(ProgressReporter):

    def __init__(self, progress_queue: Queue):
        super().__init__()
        self.queue = progress_queue

    def _create_progress(self, task: str) -> Progress:
        return SubprocessProgress(task, self.queue)

    def _create_gauge(self, name: str, low: float = None, high: float = None) -> Gauge:
        return SubprocessGauge(name, low, high, self.queue)


_PROGRESS_QUEUE = None


def _progress_queue_feed(q: Queue):
    """
    Processes progress events from the queue.
    """
    while True:
        try:
            m = q.get(True)
        except (ValueError, EOFError, BrokenPipeError):
            # queue is closed
            return
        try:
            m.apply()
        except Exception as e:
            _logger.error("processing progress", e)
        finally:
            q.task_done()


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


def setup_subprocess_progress(progress_queue: Queue, level: int):
    """
    Setup this process to send progress to the parent process.
    """
    global _subprocess_progress_configured
    if _subprocess_progress_configured:
        return
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    set_progress_reporter(SubprocessProgressReporter(progress_queue))
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
        h.close()
    logging.root.addHandler(SubprocessLogHandler(progress_queue))
    logging.root.setLevel(level)
    _subprocess_progress_configured = True


def _percent_renderer(value: float) -> str:
    return f"{value: 3.1f}%"


class ComputeGauges(object):
    def __init__(self):
        self.cpu_percent = gauge('CPU', 0, 100)
        self.cpu_percent.renderer = _percent_renderer
        self.cpu_percent.critical_range = (90, 101)

        loadavg_renderer = lambda v: f"{v: 2.2f}"
        self.loadavg1 = gauge('load1')
        self.loadavg1.renderer = loadavg_renderer
        # self.loadavg5 = gauge('load5')
        # self.loadavg5.renderer = loadavg_renderer
        # self.loadavg15 = gauge('load15')
        # self.loadavg15.renderer = loadavg_renderer

        self.mem = gauge('MEM', 0, 100)
        self.mem.renderer = _percent_renderer
        self.mem.critical_range = (90, 101)

        if sys.stdout.isatty():
            gpu_pct, gmem_pct = hwaccel.hwaccel_gpustat()
        else:
            gpu_pct, gmem_pct = None, None
        if gpu_pct is None:
            self.gpu_percent = None
        else:
            self.gpu_percent = gauge('GPU', 0, 100)
            self.gpu_percent.renderer = _percent_renderer
            self.gpu_percent.critical_range = (90, 101)
            self.gpu_percent.value(gpu_pct)

        if gmem_pct is None:
            self.gmem_percent = None
        else:
            self.gmem_percent = gauge('GPU MEM', 0, 100)
            self.gmem_percent.renderer = _percent_renderer
            self.gmem_percent.critical_range = (90, 101)
            self.gmem_percent.value(gmem_pct)

    def update(self):
        loadavg = os.getloadavg()
        self.loadavg1.value(loadavg[0])
        # self.loadavg5.value(loadavg[1])
        # self.loadavg15.value(loadavg[2])
        self.cpu_percent.value(psutil.cpu_percent(interval=None))
        self.mem.value(psutil.virtual_memory().percent)

        if self.gpu_percent is not None or self.gmem_percent is not None:
            gpu_pct, gmem_pct = hwaccel.hwaccel_gpustat()
            if gpu_pct is not None:
                self.gpu_percent.value(gpu_pct)
            if gmem_pct is not None:
                self.gmem_percent.value(gmem_pct)


class LinuxIOWaitGauge(object):
    def __init__(self):
        self.gauge = gauge('IO', 0, 100)
        self.gauge.renderer = _percent_renderer
        self.gauge.warning_range = (20, 50)
        self.gauge.critical_range = (50, 101)

        self._last_stats = self._stats()

    def _stats(self) -> Union[dict[str, int], None]:
        try:
            with open("/proc/stat", "rt") as f:
                for line in f.readlines():
                    if line.startswith("cpu "):
                        parts = line.split()
                        return {
                            'usr': int(parts[1]),
                            'nice': int(parts[2]),
                            'sys': int(parts[3]),
                            'idle': int(parts[4]),
                            'iowait': int(parts[5]),
                            'irq': int(parts[6]),
                            'softirq': int(parts[7]),
                            'steal': int(parts[8]),
                            'guest': int(parts[9]),
                        }
        except Exception as e:
            logging.root.warning("reading /proc/stat", e)
            return None

    def _total(self, stats: dict[str, int]):
        return sum(stats.values())

    def update(self):
        current_stats = self._stats()
        if current_stats is None:
            return

        if self._last_stats is not None:
            current_total = self._total(current_stats)
            last_total = self._total(self._last_stats)
            if current_total == last_total:
                return
            iowait_pct = ((current_stats['iowait'] - self._last_stats['iowait']) * 100) / (
                    current_total - last_total)
            self.gauge.value(iowait_pct)

        self._last_stats = current_stats


def start_compute_gauges(interval=30):
    gauges = ComputeGauges()
    if os.path.exists("/proc/stat"):
        io_gauge = LinuxIOWaitGauge()
    else:
        io_gauge = None

    def update():
        gauges.update()
        if io_gauge:
            io_gauge.update()
        t = threading.Timer(interval, update)
        t.daemon = True
        t.start()

    update()
