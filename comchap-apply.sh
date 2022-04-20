#!/usr/bin/env bash

# random sleep to prevent multiple containers running at the exact same time
sleep $((5 + RANDOM % 30))

exec ionice -c 3 nice -n 15 /usr/local/bin/comchap-apply.py >>/var/log/comchap.log 2>&1
