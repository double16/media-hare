#!/usr/bin/env bash

#
# Installs drivers/libraries for hardware acceleration based on what is installed on the host.
#

set -e

if [ -s '/sys/module/nvidia/version' ]; then
  NV_VERSION="$(</sys/module/nvidia/version)"
  NV_MAJOR_VER="${NV_VERSION/.*/}"
  NV_PKG_INSTALL=()
  for NV_PKG in nvidia-headless-no-dkms nvidia-utils libnvidia-decode libnvidia-encode; do
    if ! apt list "${NV_PKG}-${NV_MAJOR_VER}" 2>/dev/null | grep -q "${NV_VERSION}"; then
      NV_PKG_INSTALL+=("${NV_PKG}-${NV_MAJOR_VER}=${NV_VERSION}-*")
    fi
  done
  if [ -n "${NV_PKG_INSTALL}" ]; then
      apt-get update
      apt-get -y install "${NV_PKG_INSTALL[@]}"
  fi
fi
