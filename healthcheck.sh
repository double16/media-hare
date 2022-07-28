#!/usr/bin/env bash

# ensure cron process is running
test -f /var/run/crond.pid && test -f /proc/$(</var/run/crond.pid)/cmdline || exit 1

# ensure Xorg is running for subtitle-edit
pgrep Xorg >/dev/null || exit 1

# check logs for processes having run within 24 hours
for LOG in /var/log/transcode.log /var/log/comtune.log /var/log/profanity-filter.log; do
  if [[ -f "${LOG}" ]]; then
    AGE="$(( $(date --utc +%s) - $(stat --format=%Y ${LOG}) ))"
    if [[ ${AGE} -gt 90000 ]]; then
      exit 1
    fi
  fi
done

exit 0
