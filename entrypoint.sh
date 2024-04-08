#!/usr/bin/env bash

if [ -n "${CRON_DISABLE}" ]; then
  echo "Disabling cron and anacron because of CRON_DISABLE=${CRON_DISABLE}"
  rm /etc/cron.d/anacron
  find /usr /etc /var -name "*cron*.service" -type f -delete -print
  find /etc -name "*cron*" -type f -delete -print
fi

exec /usr/bin/systemctl default
