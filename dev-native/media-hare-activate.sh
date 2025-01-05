#!/usr/bin/env bash
script_dir=$(cd "$(dirname "$0")" && pwd)
docker compose -p mh -f "${script_dir}/docker-compose-langtools.yml" up -d
source "${script_dir}/../.venv/bin/activate"
export PATH="${PATH}:${script_dir}/../dvrprocess"
export LANGUAGE_TOOL_HOST=localhost
export KALDI_EN_HOST=localhost
export KALDI_EN_PORT=2700
export LANGUAGE_TOOL_PORT=8100
