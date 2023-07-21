import traceback

#
# User interface for terminal (curses).
#

from common import terminalui, progress
import logging
import sys
from time import sleep, time

logger = logging.getLogger(__name__)


def _example(argv: list[str]) -> int:
    logger.debug("debug message")
    logger.info("info message")
    logger.error("error message 1")
    logger.error("error message 2")
    p_list = []
    for i in range(1, 30):
        p_list.append(progress.progress(f"p{i}", 0, 100))
    for value in [25, 50, 75, 100]:
        sleep(1)
        for p in p_list:
            if p.task in ["p2", "p4", "p5"] and value >= 50:
                if value == 50:
                    p.stop()
            else:
                p.progress(value)
        sleep(0.75)
    logger.error("progress done")
    for i in range(0, 100):
        logger.error("message %i", i)
    for p in p_list:
        p.stop()
    sleep(2)
    return 0


if __name__ == '__main__':
    # this is example usage for development purposes
    try:
        sys.exit(terminalui.terminalui_wrapper(_example, sys.argv))
    except Exception as e:
        traceback.print_exception(e)
