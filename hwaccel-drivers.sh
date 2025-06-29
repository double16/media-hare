#!/usr/bin/env bash

#
# Installs drivers/libraries for hardware acceleration based on what is installed on the host.
#

set -e

APT_UPDATED=

if [[ "${HWACCEL_DRIVERS_INSTALL}" != "true" ]]; then
  exit 0
fi

if [ -s '/sys/module/nvidia/version' ]; then
  NV_VERSION="$(</sys/module/nvidia/version)"
  NV_MAJOR_VER="${NV_VERSION/.*/}"
  NV_PKG_SUFFIX="-server"
  NV_PKG_INSTALL=()
  for NV_PKG in nvidia-headless-no-dkms nvidia-utils libnvidia-decode libnvidia-encode; do
    if ! apt list --installed "${NV_PKG}-${NV_MAJOR_VER}${NV_PKG_SUFFIX}" 2>/dev/null | grep -q "${NV_VERSION}"; then
      if [[ -z "${APT_UPDATED}" ]]; then
        apt-get update
        APT_UPDATED=1
      fi
      NV_PKG_VERSION="$(apt list -a "${NV_PKG}-${NV_MAJOR_VER}${NV_PKG_SUFFIX}" 2>/dev/null | grep -F "${NV_VERSION}" | head -n 1 | tr -s '[:space:]' | cut -d ' ' -f 2)"
      if [ -n "${NV_PKG_VERSION}" ]; then
        NV_PKG_INSTALL+=("${NV_PKG}-${NV_MAJOR_VER}${NV_PKG_SUFFIX}=${NV_PKG_VERSION}")
      else
        NV_PKG_VERSION="$(apt list -a "${NV_PKG}-${NV_MAJOR_VER}" 2>/dev/null | grep -F "${NV_VERSION}" | head -n 1 | tr -s '[:space:]' | cut -d ' ' -f 2)"
        if [ -n "${NV_PKG_VERSION}" ]; then
          NV_PKG_INSTALL+=("${NV_PKG}-${NV_MAJOR_VER}=${NV_PKG_VERSION}")
        fi
      fi
    fi
  done
#  for SUPPORT_PKG in gpustat; do
#    if ! apt list --installed "${SUPPORT_PKG}" 2>/dev/null | grep -q "${SUPPORT_PKG}"; then
#      if [[ -z "${APT_UPDATED}" ]]; then
#        apt-get update
#        APT_UPDATED=1
#      fi
#      NV_PKG_INSTALL+=("${SUPPORT_PKG}")
#    fi
#  done
  if [ -n "${NV_PKG_INSTALL}" ]; then
      apt-get -y install "${NV_PKG_INSTALL[@]}"
  fi
fi
