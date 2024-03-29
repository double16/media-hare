#!/usr/bin/env bash

# random sleep to prevent multiple programs running at the exact same time
sleep $((5 + RANDOM % 30))

exec ionice -c 3 nice -n 15 /usr/local/bin/profanity-filter-apply.py --ignore-compute -u . "$@" >>/var/log/profanity-filter.log 2>&1
