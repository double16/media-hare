#!/usr/bin/env bash

# random sleep to prevent multiple programs running at the exact same time
sleep $((5 + RANDOM % 30))

exec ionice -c 3 nice -n 15 /usr/local/bin/transcode-apply.py "$@" >>/var/log/transcode.log 2>&1
