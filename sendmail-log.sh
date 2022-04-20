#!/bin/sh

# Logs sendmail calls

{
  date
  echo
  echo "$@"
  echo
  cat
} >> /var/log/sendmail.log
