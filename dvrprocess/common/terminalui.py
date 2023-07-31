
#
# User interface for terminal (curses).
#

import curses
import logging
import re
import time
from math import ceil, floor
from typing import Union, Dict

from . import progress
from .proc_invoker import StreamCapture

logger = logging.getLogger(__name__)


LOG_MSG_CLEAN = re.compile("[\r\n]+")


_CURSESUI = None


def _check_resize():
    c = _CURSESUI.screen.getch()
    if c == curses.KEY_RESIZE:
        _CURSESUI.resize()


class CursesLogHandler(logging.Handler):
    def __init__(self, pad, window, level=logging.INFO):
        super().__init__(level)
        self.pad = pad
        self.window = window
        self.y = 0

    def emit(self, record: logging.LogRecord) -> None:
        _check_resize()
        try:
            maxy, maxx = self.pad.getmaxyx()
            if self.y >= maxy:
                self.pad.move(0, 0)
                self.pad.insdelln(-1)
                self.y = maxy - 1

            created_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
            msg = f"{created_str} {record.levelname:<5} {record.filename:<10}  {record.getMessage()}"

            attr = curses.A_NORMAL
            if record.levelno >= logging.ERROR:
                attr = curses.A_STANDOUT
            self.pad.insnstr(self.y, 0, msg, maxx-1, attr)
            self.y = min(self.y+1, maxy)

            self.refresh()
        except:
            # don't break the application because of a logging error
            pass

    def refresh(self):
        win_top, win_left = self.window.getbegyx()
        win_top += 1
        win_left += 1
        win_height, win_width = self.window.getmaxyx()
        win_bottom = win_top + win_height - 3
        win_right = win_left + win_width - 3
        self.pad.refresh(max(0, self.y - win_height + 2), 0, win_top, win_left, win_bottom, win_right)

    def resize(self):
        self.window.clear()
        self.window.border()
        self.window.refresh()
        self.refresh()


class CursesProgressListener(object):
    def start(self, task):
        pass

    def progress(self, task, position: int, msg: str):
        pass

    def stop(self, task):
        pass


class ProgressCurses(progress.Progress):

    def __init__(self, task: str, listener: CursesProgressListener):
        super().__init__(task)
        self.listener = listener
        self.relative_row = -1  # not displayed

    def start(self, start: int, end: int, msg: Union[str, None] = None) -> None:
        super().start(start, end, msg)
        self.listener.start(self)
        if end > start:
            self.progress(start, msg, start, end)

    def stop(self, msg: Union[str, None] = None) -> None:
        super().stop(msg)
        self.listener.stop(self)

    def progress(self, position: int, msg: Union[str, None] = None, start: Union[int, None] = None, end: Union[int, None] = None) -> None:
        super().progress(position, msg, start, end)
        if self.update_reporting():
            self.listener.progress(self, position, msg)


class ProgressWindow(CursesProgressListener):

    def __init__(self, window):
        self.window = window
        self.tasks: Dict[str, ProgressCurses] = dict()

    def refresh(self):
        self.window.refresh()

    def resize(self):
        lines, cols = self.window.getmaxyx()
        for task in list(self.tasks.values()):
            if task.relative_row >= lines:
                task.relative_row = -1
            else:
                self._draw(task, task.last_position)

    def _allocate_row(self, task: ProgressCurses):
        if task.relative_row >= 0:
            return
        # look for unused
        tasks = list(self.tasks.values())
        taken_rows = set([t.relative_row for t in tasks])
        for row in range(0, self.window.getmaxyx()[0]):
            if row in taken_rows:
                continue
            task.relative_row = row
            return
        # replace older task
        if tasks:
            tasks.sort(key=lambda e: e.get_last_progress_time() or 0, reverse=True)
            oldest_task = tasks[0]
            self.tasks.pop(oldest_task.task, None)
            task.relative_row = oldest_task.relative_row
            oldest_task.relative_row = -1

    def _draw(self, task: ProgressCurses, position: int = 0, msg: str = None):
        if task.relative_row < 0:
            return
        lines, cols = self.window.getmaxyx()
        if task.relative_row >= lines:
            # resize event occurred
            task.relative_row = -1
            return
        if cols > 12:
            if cols > 20:
                eta_width = 9
            else:
                eta_width = 0
            bar_width = max(8, ceil(cols * 0.3))
            msg_width = cols - bar_width - eta_width - 2

            if msg:
                label = "%s - %s" % (task.task, msg)
            else:
                label = task.task

            if task.pct is None:
                bar_complete = ""
            else:
                bar_complete = "=" * ceil((bar_width-2)*(task.pct/100))
            bar_incomplete = " " * (bar_width - 2 - len(bar_complete))
            bar = f"[{bar_complete}{bar_incomplete}]"

            position_str = task.position_str(position)
            if position_str:
                bar_left = max(1, floor((bar_width - len(position_str))/2))
                bar_right = min(len(bar)-1, bar_left + 2 + len(position_str))
                bar = f"{bar[0:bar_left]} {position_str} {bar[bar_right:]}"

            output = str(f"{label[0:msg_width+1]:>{msg_width}} {bar:<{bar_width}}")
            if eta_width > 0 and task.eta is not None:
                output = f"{output} {task.remaining_human_duration():>{eta_width-1}}"
        else:
            output = ""
        self.window.move(task.relative_row, 0)
        self.window.addstr(output)
        self.window.clrtoeol()

    def start(self, task: ProgressCurses):
        _check_resize()
        existing_task = self.tasks.get(task.task, None)
        if existing_task:
            task.relative_row = existing_task.relative_row
        else:
            self._allocate_row(task)
        self.tasks[task.task] = task
        if task.relative_row >= 0:
            self._draw(task)
            self.refresh()

    def progress(self, task: ProgressCurses, position: int, msg: str):
        _check_resize()
        if task.relative_row < 0:
            self._allocate_row(task)
            if task.relative_row < 0:
                return
        self._draw(task, position, msg)
        self.refresh()

    def stop(self, task: ProgressCurses):
        _check_resize()
        self.tasks.pop(task.task, None)
        if task.relative_row >= 0:
            self.window.move(task.relative_row, 0)
            self.window.clrtoeol()
            self.refresh()


class CursesGaugeListener(object):
    def create(self, gauge):
        pass

    def value(self, gauge, value: float):
        pass


class GaugeCurses(progress.Gauge):
    def __init__(self, name: str, low: float, high: float, listener: CursesGaugeListener):
        super().__init__(name, low, high)
        self.listener = listener
        self.listener.create(self)

    def value(self, value: float):
        super().value(value)
        if self.update_reporting():
            self.listener.value(self, value)


class GaugeWindow(CursesGaugeListener):
    def __init__(self, window):
        self.window = window
        self.gauges: Dict[str, GaugeCurses] = dict()

    def resize(self):
        self.window.erase()
        self.window.move(self.window.getbegyx()[0], self.window.getbegyx()[1])
        try:
            for idx, gauge in enumerate(self.gauges.values()):
                if idx > 0:
                    self.window.addstr(" | ")
                self.window.addstr(gauge.name)
                self.window.addstr(" ")
                attr = curses.A_NORMAL
                if gauge.is_critical(gauge.last_value):
                    attr = curses.A_STANDOUT
                self.window.addstr(gauge.value_str(gauge.last_value), attr)
        except curses.error:
            # outside the window
            pass

        self.window.refresh()

    def create(self, gauge):
        self.gauges[gauge.name] = gauge
        self.resize()

    def value(self, gauge, value: float):
        self.resize()


class CursesProgressReporter(progress.ProgressReporter):
    def __init__(self, progress_window: ProgressWindow, gauge_window: GaugeWindow):
        super().__init__()
        self.progress_window = progress_window
        self.gauge_window = gauge_window

    def _create_progress(self, task: str) -> progress.Progress:
        return ProgressCurses(task, self.progress_window)

    def _create_gauge(self, name: str, low: float = None, high: float = None) -> progress.Gauge:
        return GaugeCurses(name, low, high, self.gauge_window)


class CursesUI(object):
    def __init__(self, screen):
        # hide the cursor
        try:
            curses.curs_set(0)
        except:
            pass

        self.gauge_win: GaugeWindow = None
        self.progress_win: ProgressWindow = None
        self.log_handler: CursesLogHandler = None
        self.screen = screen
        self.screen.nodelay(True)
        window_dims = self._compute_window_dims()
        self.gauge_win = GaugeWindow(curses.newwin(*window_dims[0]))

        log_pad = curses.newpad(1000, 500)
        log_win = curses.newwin(*window_dims[2])
        self.log_handler = CursesLogHandler(log_pad, log_win)
        logging.root.addHandler(self.log_handler)
        logging.root.setLevel(logging.INFO)

        self.progress_win = ProgressWindow(curses.newwin(*window_dims[1]))
        progress.set_progress_reporter(CursesProgressReporter(self.progress_win, self.gauge_win))

        self.gauge_win.resize()
        self.log_handler.resize()

    def _compute_window_dims(self) -> list[tuple[int, int, int, int]]:
        """
        :return: tuples of (lines, columns, y, x): [ status, progress, log ]
        """
        lines, cols = self.screen.getmaxyx()
        status_win = (1, cols, 0, 0)
        progress_win = (ceil(lines/3), cols, 1, 0)
        log_win = (lines - 1 - progress_win[0], cols, progress_win[0] + progress_win[2], 0)
        return [status_win, progress_win, log_win]

    def resize(self):
        self.screen.refresh()
        window_dims = self._compute_window_dims()
        self._resize_window(self.gauge_win.window, window_dims[0])
        self.gauge_win.resize()
        self._resize_window(self.progress_win.window, window_dims[1])
        self.progress_win.resize()
        self._resize_window(self.log_handler.window, window_dims[2])
        self.log_handler.resize()

    def _resize_window(self, window, dims: tuple[int, int, int, int]):
        window.resize(dims[0], dims[1])
        window.mvwin(dims[2], dims[3])


def terminalui_wrapper(func, *args, **kwargs) -> int:
    """
    :param func:
    :param args:
    :param kwargs:
    :return: return code for sys.exit
    """
    def main(screen) -> int:
        global _CURSESUI
        screen.refresh()
        _CURSESUI = CursesUI(screen)
        progress.start_compute_gauges(2)

        return func(*args, **kwargs)

    stderr_capture = StreamCapture('stderr')
    stdout_capture = StreamCapture('stdout')
    try:
        return curses.wrapper(main)
    finally:
        stderr_capture.finish()
        stdout_capture.finish(output=False)
