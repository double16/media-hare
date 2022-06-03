#!/usr/bin/env bash

#
# SubtitleEdit 3.6.x works best with tesseract 3. Ubuntu has v4. This script attempts to adjust tesseract options
# to more closely align with SE 3.6 expectations.
#

REAL_TESSERACT="/usr/bin/tesseract.orig"
declare -a ENGINE_OPTS ORIG_OPTS

ORIG_OPTS=("$@")
OEM=3

function finish() {
    if [[ -z "${COMPLETE}" ]] || [[ -n "${ABORT}" ]] || [[ -z "${INJECTED}" ]]; then
        echo "${REAL_TESSERACT}" --oem ${OEM} "${ORIG_OPTS[@]}" >> /var/log/tesseract.log 2>&1
        exec "${REAL_TESSERACT}" --oem ${OEM} "${ORIG_OPTS[@]}" >> /var/log/tesseract.log 2>&1
    else
        echo "${REAL_TESSERACT}" "${ENGINE_OPTS[@]}" >> /var/log/tesseract.log 2>&1
        exec "${REAL_TESSERACT}" "${ENGINE_OPTS[@]}" >> /var/log/tesseract.log 2>&1
    fi
}
trap finish EXIT

# Check version
VERSION="$("${REAL_TESSERACT}" --version 2>&1 | head -n 1 | cut -d ' ' -f 2)"
if [[ "${VERSION}" =~ ^3[.].* ]]; then
    exec "${REAL_TESSERACT}" "$@" >> /var/log/tesseract.log 2>&1
fi

# Flag to mark than we found options we are not compatible with
ABORT=
# Flag to mark that we injected engine, so we should use our options
INJECTED=

for OPT in "$@"; do
    SKIP_OPT=
    if [[ -n "${INJECTED}" ]]; then
        # We've already injected options, don't process any longer
        true
    elif [[ "${LAST}" =~ --oem ]]; then
        #  0    Legacy engine only.
        #  1    Neural nets LSTM engine only.
        #  2    Legacy + LSTM engines.
        #  3    Default, based on what is available, which is related to the installed models.
        ENGINE_OPTS+=("${OEM}")
        SKIP_OPT=1
        INJECTED=1
    fi

    if [[ -f "${OPT}" ]]; then
      if identify -verbose "${OPT}" | grep -i transpar | grep -v none | grep -q .; then
        echo "Transparency found" >> /var/log/tesseract.log
        identify -verbose "${OPT}" >> /var/log/tesseract.log 2>&1
        exit 255
      fi
    fi

    LAST="${OPT}"
    if [[ -z "${SKIP_OPT}" ]]; then
        ENGINE_OPTS+=("${OPT}")
    fi
done
COMPLETE=1
