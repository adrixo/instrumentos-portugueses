#!/usr/bin/env bash
# Entrypoint del contenedor GPU (ADR §8.1: fijar env de determinismo antes de torch).
set -euo pipefail

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec "$@"
