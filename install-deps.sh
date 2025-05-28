#!/usr/bin/env bash

mkdir -p /var/cache/pip

pip --cache-dir /var/cache/pip \
    --no-input install \
    --break-system-packages \
    --compile --ignore-installed \
    -r /usr/local/share/requirements.txt

if test -f /usr/local/lib/python3.12/dist-packages/whisper/triton_ops.py && test -f /usr/local/src/whisper.patch; then
  pushd /usr/local/lib/python3.12/dist-packages
  patch -f -p0 < /usr/local/src/whisper.patch || true
  popd
fi

python3 -c "import whisper; whisper.load_model('${WHISPER_MODEL}')"
