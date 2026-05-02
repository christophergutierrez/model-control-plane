#!/usr/bin/env bash
set -euo pipefail

VLLM_VENV="/home/chris/vllm-install/.vllm"
REGISTRY_ROOT="/home/chris/models"
ANCHOR_ROUTE="videoamp/api/programs"
ROLE="responder"
HOST="127.0.0.1"
PORT="8000"
DTYPE="bfloat16"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME="/home/chris/git_home/trainLLM/models/hf"

source "${VLLM_VENV}/bin/activate"

exec python3 "${SCRIPT_DIR}/serve_vllm.py" \
    "${REGISTRY_ROOT}" \
    "${ANCHOR_ROUTE}" \
    --role "${ROLE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    -- --enforce-eager "$@"
