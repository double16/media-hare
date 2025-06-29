#
# User interface for terminal (curses).
#
import atexit
import curses
import logging
import re
import threading
import time
from math import ceil, floor
from typing import Union, Dict

from . import progress
from .proc_invoker import StreamCapturingLogger, pre_flight_check

logger = logging.getLogger(__name__)

LOG_MSG_CLEAN = re.compile("[\r\n]+")

_CURSESUI = None
_CURSESUI_LAST_RESIZE = 0
_curses_ui_lock = threading.RLock()


def _check_resize():
    global _CURSESUI, _CURSESUI_LAST_RESIZE
    c = _CURSESUI.screen.getch()
    if c == curses.KEY_RESIZE or (time.time() - _CURSESUI_LAST_RESIZE > 300):
        _CURSESUI_LAST_RESIZE = time.time()
        _CURSESUI.resize()


class CursesLogHandler(logging.Handler):
    log_msg_cleaner = re.compile(r'[\u0000-\u001F\u007F-\uFFFF]+')

    def __init__(self, pad, window, level=logging.INFO):
        super().__init__(level)
        self.pad = pad
        self.window = window
        self.y = 0
        self.log_records_on_close = []

        def flush_logs():
            logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s', level=logging.INFO,
                                force=True)
            summary_logger = logging.getLogger('summary')
            for record in self.log_records_on_close:
                summary_logger.handle(record)

        atexit.register(flush_logs)

    def emit(self, record: logging.LogRecord) -> None:
        _check_resize()
        try:
            _curses_ui_lock.acquire()
            max_y, max_x = self.pad.getmaxyx()
            if self.y >= max_y:
                self.pad.move(0, 0)
                self.pad.insdelln(-1)
                self.y = max_y - 1

            created_str = time.strftime('%H:%M:%S', time.localtime(record.created))
            log_msg = self.log_msg_cleaner.sub(" ", record.getMessage())
            msg = f"{created_str} {record.levelname:<7} {record.filename[:20]:<20}  {log_msg}"

            attr = curses.A_NORMAL
            if record.levelno >= logging.ERROR:
                attr = curses.A_STANDOUT
                self.log_records_on_close.append(record)
                if len(self.log_records_on_close) > 50:
                    self.log_records_on_close.pop(0)
            self.pad.insnstr(self.y, 0, msg, max_x - 1, attr)
            self.y = min(self.y + 1, max_y)

            self.refresh()
        except Exception:
            # don't break the application because of a logging error
            pass
        finally:
            _curses_ui_lock.release()

    def refresh(self):
        try:
            _curses_ui_lock.acquire()
            win_top, win_left = self.window.getbegyx()
            win_top += 1
            win_left += 1
            win_height, win_width = self.window.getmaxyx()
            win_bottom = win_top + win_height - 3
            win_right = win_left + win_width - 3
            self.pad.refresh(max(0, self.y - win_height + 2), 0, win_top, win_left, win_bottom, win_right)
        finally:
            _curses_ui_lock.release()

    def resize(self):
        try:
            _curses_ui_lock.acquire()
            self.window.clear()
            self.window.border()
            self.window.refresh()
        finally:
            _curses_ui_lock.release()
        self.refresh()


class CursesProgressListener(object):
    def start(self, task):
        pass

    def progress(self, task, position: int, msg: str):
        pass

    def stop(self, task):
        pass

    def close(self):
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

    def progress(self, position: int, msg: Union[str, None] = None, start: Union[int, None] = None,
                 end: Union[int, None] = None) -> None:
        super().progress(position, msg, start, end)
        if self.update_reporting():
            self.listener.progress(self, position, msg)


class ProgressWindow(CursesProgressListener):

    def __init__(self, window):
        self._closed = False
        self.window = window
        self.tasks: Dict[str, ProgressCurses] = dict()

    def close(self):
        self._closed = True

    def refresh(self):
        if self._closed:
            return

        self.window.refresh()

    def resize(self):
        if self._closed:
            return

        try:
            _curses_ui_lock.acquire()
            lines, cols = self.window.getmaxyx()
            for task in list(self.tasks.values()):
                if task.relative_row >= lines:
                    task.relative_row = -1
                else:
                    self._draw(task, task.last_position)
        finally:
            _curses_ui_lock.release()

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
                bar_complete = "=" * ceil((bar_width - 2) * (min(task.pct, 100) / 100))
            bar_incomplete = " " * (bar_width - 2 - len(bar_complete))
            bar = f"[{bar_complete}{bar_incomplete}]"

            position_str = task.position_str(position)
            if position_str:
                bar_left = max(1, floor((bar_width - len(position_str)) / 2))
                bar_right = min(len(bar) - 1, bar_left + 2 + len(position_str))
                bar = f"{bar[0:bar_left]} {position_str} {bar[bar_right:]}"

            output = str(f"{label[0:msg_width + 1]:>{msg_width}} {bar:<{bar_width}}")
            if eta_width > 0 and task.eta is not None:
                output = f"{output} {task.remaining_human_duration():>{eta_width - 1}}"
        else:
            output = ""
        self.window.move(task.relative_row, 0)
        self.window.addstr(output)
        self.window.clrtoeol()

    def start(self, task: ProgressCurses):
        if self._closed:
            return

        _check_resize()
        try:
            _curses_ui_lock.acquire()
            existing_task = self.tasks.get(task.task, None)
            if existing_task:
                task.relative_row = existing_task.relative_row
            else:
                self._allocate_row(task)
            self.tasks[task.task] = task
            if task.relative_row >= 0:
                self._draw(task)
                self.refresh()
        finally:
            _curses_ui_lock.release()

    def progress(self, task: ProgressCurses, position: int, msg: str):
        if self._closed:
            return

        _check_resize()
        try:
            _curses_ui_lock.acquire()
            if task.relative_row < 0:
                self._allocate_row(task)
                if task.relative_row < 0:
                    return
            self._draw(task, position, msg)
            self.refresh()
        finally:
            _curses_ui_lock.release()

    def stop(self, task: ProgressCurses):
        if self._closed:
            return

        _check_resize()
        try:
            _curses_ui_lock.acquire()
            self.tasks.pop(task.task, None)
            if task.relative_row >= 0:
                self.window.move(task.relative_row, 0)
                self.window.clrtoeol()
                self.refresh()
        finally:
            _curses_ui_lock.release()


class CursesGaugeListener(object):
    def create(self, gauge):
        pass

    def value(self, gauge, value: float):
        pass

    def close(self):
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
        self._closed = False
        self.window = window
        self.gauges: Dict[str, GaugeCurses] = dict()

    def close(self):
        self._closed = True

    def resize(self):
        if self._closed:
            return

        self.window.erase()
        self.window.move(self.window.getbegyx()[0], self.window.getbegyx()[1])
        try:
            _curses_ui_lock.acquire()
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
        finally:
            _curses_ui_lock.release()

        self.window.refresh()

    def create(self, gauge):
        if self._closed:
            return

        self.gauges[gauge.name] = gauge
        self.resize()

    def value(self, gauge, value: float):
        if self._closed:
            return

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
        self._closed = False

        # hide the cursor
        try:
            curses.curs_set(0)
        except Exception:
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
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
            h.close()
        logging.root.setLevel(logging.INFO)
        self.log_handler = CursesLogHandler(log_pad, log_win)
        logging.root.addHandler(self.log_handler)

        self.progress_win = ProgressWindow(curses.newwin(*window_dims[1]))
        progress.set_progress_reporter(CursesProgressReporter(self.progress_win, self.gauge_win))

        self.gauge_win.resize()
        self.log_handler.resize()

    def close(self):
        if self._closed:
            return

        self._closed = True
        logging.root.removeHandler(self.log_handler)
        self.log_handler.close()
        progress.set_progress_reporter(None)
        self.progress_win.close()
        self.gauge_win.close()

        # show the cursor
        try:
            curses.curs_set(1)
        except Exception:
            pass

    def _compute_window_dims(self) -> list[tuple[int, int, int, int]]:
        """
        :return: tuples of (lines, columns, y, x): [ status, progress, log ]
        """
        lines, cols = self.screen.getmaxyx()
        status_win = (1, cols, 0, 0)
        progress_win = (ceil(lines / 3), cols, 1, 0)
        log_win = (lines - 1 - progress_win[0], cols, progress_win[0] + progress_win[2], 0)
        return [status_win, progress_win, log_win]

    def resize(self):
        if self._closed:
            return

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
    pre_flight_check()

    def main(screen) -> int:
        global _CURSESUI, _CURSESUI_LAST_RESIZE
        screen.refresh()
        _CURSESUI = CursesUI(screen)
        _CURSESUI_LAST_RESIZE = time.time()
        progress.start_compute_gauges(2)

        try:
            return func(*args, **kwargs)
        finally:
            # show the cursor
            _CURSESUI.close()
            _CURSESUI = None

    stderr_capture = StreamCapturingLogger('stderr')
    stdout_capture = StreamCapturingLogger('stdout')
    try:
        return curses.wrapper(main)
    finally:
        stderr_capture.finish()
        stdout_capture.finish(output=False)
