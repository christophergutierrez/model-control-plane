#!/usr/bin/env bash
set -euo pipefail

VLLM_VENV="/home/chris/vllm-install/.vllm"
REGISTRY_ROOT="/home/chris/models"
PORT="8080"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME="/home/chris/git_home/trainLLM/models/hf"

source "${VLLM_VENV}/bin/activate"

exec python3 "${SCRIPT_DIR}/orchestrate.py" \
    "${REGISTRY_ROOT}" \
    --serve \
    --serve-port "${PORT}" \
    "$@"
