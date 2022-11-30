#!/usr/bin/env python3
import sys

import common
from common import proc_invoker

if __name__ == '__main__':
    common.setup_cli()
    if proc_invoker.pre_flight_check():
        sys.exit(0)
    else:
        sys.exit(255)
