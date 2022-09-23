#!/usr/bin/env bash

set -e

[ -z "${TVSHOW_SUMMARY_PATH}" ] && exit 0

# random sleep to prevent multiple containers running at the exact same time
sleep $((5 + RANDOM % 30))

mkdir -p "${TVSHOW_SUMMARY_PATH}"
ionice -c 3 nice -n 15 /usr/local/bin/tvshow-summary.py -e "${TVSHOW_SUMMARY_PATH}"/tvshow-episodes.csv -c "${TVSHOW_SUMMARY_PATH}"/tvshow-completion.csv >>/var/log/tvshow-summary.log 2>&1
chown --reference="${TVSHOW_SUMMARY_PATH}" "${TVSHOW_SUMMARY_PATH}"/tvshow-episodes.csv "${TVSHOW_SUMMARY_PATH}"/tvshow-completion.csv
