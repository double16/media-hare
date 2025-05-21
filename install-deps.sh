#!/usr/bin/env bash

mkdir -p /var/cache/pip

pip --cache-dir /var/cache/pip \
    --no-input install \
    --break-system-packages \
    --compile --ignore-installed \
    -r /usr/local/share/requirements.txt

python3 -c "import whisper; whisper.load_model('${WHISPER_MODEL}')"
