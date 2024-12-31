#!/usr/bin/env bash
docker compose -p mh -f "$(dirname "$0")/docker-compose-langtools.yml" up -d
source "$(dirname "$0")/../.venv/bin/activate"
export PATH="${PATH}:$(pwd)/$(dirname "$0")/../dvrprocess"
export LANGUAGE_TOOL_HOST=localhost
export KALDI_EN_HOST=localhost
export KALDI_EN_PORT=2700
export LANGUAGE_TOOL_PORT=8100
