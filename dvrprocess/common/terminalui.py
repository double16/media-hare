
#
# User interface for terminal (curses).
#

import curses
import logging
import re
from math import ceil
from typing import Union, Dict

from . import progress

logger = logging.getLogger(__name__)


LOG_MSG_CLEAN = re.compile("[\r\n]+")


class CursesLogHandler(logging.Handler):
    def __init__(self, pad, window, level=logging.INFO):
        super().__init__(level)
        self.pad = pad
        self.window = window
        self.y = 0

    def emit(self, record: logging.LogRecord) -> None:
        maxy, maxx = self.pad.getmaxyx()
        if self.y >= maxy:
            self.pad.move(self.y, 0)
            self.pad.insdelln(-1)
            self.y = maxy
        win_top, win_left = self.window.getbegyx()
        win_top += 1
        win_left += 1
        win_height, win_width = self.window.getmaxyx()
        win_bottom = win_top + win_height - 3
        win_right = win_left + win_width - 3

        try:
            msg = LOG_MSG_CLEAN.sub("", record.msg % record.args)
        except:
            msg = LOG_MSG_CLEAN.sub("", record.msg)

        self.pad.insnstr(self.y, 0, msg, maxx-1)
        self.pad.refresh(max(0, self.y - win_height), 0, win_top, win_left, win_bottom, win_right)
        self.y = min(self.y+1, maxy)


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
    tasks: Dict[str, ProgressCurses] = dict()

    def __init__(self, window):
        self.window = window

    def refresh(self):
        self.window.refresh()

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
            tasks.sort(key=lambda e: e.get_last_progress_time(), reverse=True)
            oldest_task = tasks[0]
            self.tasks.pop(oldest_task.task, None)
            task.relative_row = oldest_task.relative_row
            oldest_task.relative_row = -1

    def _draw(self, task: ProgressCurses, position: int = 0, msg: str = None):
        if task.relative_row < 0:
            return
        pct = "??" if task.pct is None else task.pct
        if msg:
            output = "%s %s%% %s - %s" % (task.task, pct, task.remaining_human_duration(), msg)
        else:
            output = "%s %s%% %s" % (task.task, pct, task.remaining_human_duration())
        self.window.move(task.relative_row, 0)
        self.window.addstr(output)
        self.window.clrtoeol()

    def start(self, task: ProgressCurses):
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
        if task.relative_row < 0:
            self._allocate_row(task)
            if task.relative_row < 0:
                return
        self._draw(task, position, msg)
        self.refresh()

    def stop(self, task: ProgressCurses):
        self.tasks.pop(task.task, None)
        if task.relative_row >= 0:
            self.window.move(task.relative_row, 0)
            self.window.clrtoeol()
            self.refresh()


class CursesProgressReporter(progress.ProgressReporter):
    def __init__(self, window: ProgressWindow):
        super().__init__()
        self.progress_win = window

    def _create_progress(self, task: str) -> progress.Progress:
        return ProgressCurses(task, self.progress_win)


class CursesUI(object):
    stat_win = None
    progress_win = None
    log_win = None
    log_pad = None

    def __init__(self, screen):
        # hide the cursor
        try:
            curses.curs_set(0)
        except:
            pass

        self.screen = screen
        self.log_pad = curses.newpad(1000, 500)
        window_dims = self._compute_window_dims()
        self.stat_win = curses.newwin(*window_dims[0])
        self.progress_win = ProgressWindow(curses.newwin(*window_dims[1]))
        self.log_win = curses.newwin(*window_dims[2])

        logging.root.addHandler(CursesLogHandler(self.log_pad, self.log_win))
        logging.root.setLevel(logging.INFO)
        progress.set_progress_reporter(CursesProgressReporter(self.progress_win))

        self.stat_win.addstr("CPU %  GPU %  MEM %")
        self.stat_win.refresh()
        self.log_win.border()
        self.log_win.refresh()

    def _compute_window_dims(self):
        """
        :return: tuples of (lines, columns, y, x): [ status, progress, log ]
        """
        lines, cols = self.screen.getmaxyx()
        status_win = (1, cols, 0, 0)
        progress_win = (ceil(lines/2), cols, 1, 0)
        log_win = (lines - 1 - progress_win[0], cols, progress_win[0] + progress_win[2], 0)
        return [status_win, progress_win, log_win]


def terminalui_wrapper(func, *args, **kwargs) -> int:
    """
    :param func:
    :param args:
    :param kwargs:
    :return: return code for sys.exit
    """
    def main(screen) -> int:
        screen.refresh()
        CursesUI(screen)

        return func(*args, **kwargs)

    return curses.wrapper(main)
