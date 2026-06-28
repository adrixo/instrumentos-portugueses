#!/usr/bin/env bash
# Comprobación de GPU/CUDA (ADR §11.3).
set -euo pipefail

echo "== nvidia-smi =="
nvidia-smi || echo "nvidia-smi no disponible"

echo "== torch =="
python -c "import torch; print('cuda available:', torch.cuda.is_available()); \
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
