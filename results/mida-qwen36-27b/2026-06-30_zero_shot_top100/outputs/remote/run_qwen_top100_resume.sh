#!/usr/bin/env bash
set -euo pipefail
cd /home/esalab/Escritorio/instrumentos_portugueses_ir
docker run --rm --gpus all --ipc=host --entrypoint /bin/bash   -e NVIDIA_VISIBLE_DEVICES=all   -e HF_HOME=/models/huggingface   -e TRANSFORMERS_CACHE=/models/huggingface   -e MLFLOW_TRACKING_URI=sqlite:////workspace/outputs/mlflow.db   -e VLM_DISABLE_THINKING=true   -e VLM_CACHE=1   -e VLM_CACHE_DIR=outputs/cache/vlm_openai   -e VLM_MAX_IMAGE_SIDE=768   -e VLM_JPEG_QUALITY=85   -v /home/esalab/Escritorio/instrumentos_portugueses_ir:/workspace   -v /home/esalab/Escritorio/instrumentos_portugueses_ir/models:/models   -w /workspace   docker-irlab:latest -lc '
set -euo pipefail
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
python scripts/qwen36_top100_resume.py   --dense-run outputs/runs/DENSE_qwen36_top100_test.trec   --previous-trace outputs/rerank_traces/B4_qwen36_zero_shot_test.jsonl   --run-name B4_qwen36_zero_shot_top100_test   --top-n 100 --final-top-k 100   --base-url http://100.127.120.42:8080/v1   --vlm-model qwen36-27b   --timeout 120
instrument-ir evaluate --run outputs/runs/B4_qwen36_zero_shot_top100_test.trec --qrels data/processed/qrels/test.qrels
instrument-ir rerank-metrics   --dense-run outputs/runs/DENSE_qwen36_top100_test.trec   --reranked-run outputs/runs/B4_qwen36_zero_shot_top100_test.trec   --qrels data/processed/qrels/test.qrels   --n 100 --k 100   --out outputs/metrics/B4_qwen36_zero_shot_top100_test__rerankmetrics.json
instrument-ir report
'
