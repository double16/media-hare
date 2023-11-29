import linecache
import os
import sys
import tracemalloc
from datetime import datetime
from queue import Queue, Empty
from resource import getrusage, RUSAGE_SELF
from threading import Thread
from typing import Union

from . import config

_QUEUE = Queue()
_MONITOR_THREAD: Union[Thread, None] = None


def memory_monitor(command_queue: Queue = _QUEUE, poll_interval=1):
    tracemalloc.start()
    old_max = 0
    snapshot = None
    while True:
        try:
            command_queue.get(timeout=poll_interval)
            if snapshot is not None:
                print(datetime.now(), 'PID', os.getpid(), file=sys.stderr)
                display_top(snapshot)

            return
        except Empty:
            max_rss = getrusage(RUSAGE_SELF).ru_maxrss
            if max_rss > old_max:
                old_max = max_rss
                snapshot = tracemalloc.take_snapshot()
                print(datetime.now(), 'PID', os.getpid(), 'max RSS', config.bytes_to_human_str(max_rss), file=sys.stderr)


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit, file=sys.stderr)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024), file=sys.stderr)
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line, file=sys.stderr)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024), file=sys.stderr)
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024), file=sys.stderr)


def memory_monitor_start():
    global _MONITOR_THREAD
    if os.getenv('MEMORY_MONITOR', '') and _MONITOR_THREAD is None:
        _MONITOR_THREAD = Thread(target=memory_monitor)
        _MONITOR_THREAD.start()


def memory_monitor_stop():
    global _MONITOR_THREAD, _QUEUE
    if _MONITOR_THREAD:
        _QUEUE.put('stop')
        _MONITOR_THREAD.join()
        _MONITOR_THREAD = None
