#!/usr/bin/env bash
# Precarga de modelos en la caché de HuggingFace (ADR §9). Ejecutar una vez en la máquina GPU.
set -euo pipefail
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-VL-3B-Instruct", "vidore/colqwen2-v1.0", "jinaai/jina-clip-v2"]:
    print("descargando", m)
    snapshot_download(m)
PY
