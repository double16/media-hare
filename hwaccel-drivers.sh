#!/usr/bin/env bash

#
# Installs drivers/libraries for hardware acceleration based on what is installed on the host.
#

set -e

APT_UPDATED=

if [ -s '/sys/module/nvidia/version' ]; then
  NV_VERSION="$(</sys/module/nvidia/version)"
  NV_MAJOR_VER="${NV_VERSION/.*/}"
  NV_PKG_INSTALL=()
  for NV_PKG in nvidia-headless-no-dkms nvidia-utils libnvidia-decode libnvidia-encode; do
    if ! apt list --installed "${NV_PKG}-${NV_MAJOR_VER}" 2>/dev/null | grep -q "${NV_VERSION}"; then
      if [[ -z "${APT_UPDATED}" ]]; then
        apt-get update
        APT_UPDATED=1
      fi
      NV_PKG_VERSION="$(apt list -a "${NV_PKG}-${NV_MAJOR_VER}" 2>/dev/null | grep -F "${NV_VERSION}" | head -n 1 | tr -s '[:space:]' | cut -d ' ' -f 2)"
      NV_PKG_INSTALL+=("${NV_PKG}-${NV_MAJOR_VER}=${NV_PKG_VERSION}")
    fi
  done
  if [ -n "${NV_PKG_INSTALL}" ]; then
      apt-get -y install "${NV_PKG_INSTALL[@]}"
  fi
fi
