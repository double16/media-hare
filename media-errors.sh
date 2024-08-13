#!/usr/bin/env bash

# random sleep to prevent multiple programs running at the exact same time
sleep $((5 + RANDOM % 30))

exec ionice -c 3 nice -n 15 /usr/local/bin/find_media_errors.py "$@" >>/var/log/find_media_errors.log 2>&1
