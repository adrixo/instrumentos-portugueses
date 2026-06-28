#!/usr/bin/env bash
# Pipeline final en test (ADR §17.2). Requiere GPU + servidor VLM para B4/B5.
set -euo pipefail
DENSE=outputs/runs/B1_openclip-vitb32_test.trec
instrument-ir build-qrels --split test
instrument-ir retrieve --split test --model openclip-vitb32 --run-name B1_openclip-vitb32_test
instrument-ir evaluate --run "$DENSE" --qrels data/processed/qrels/test.qrels
instrument-ir rerank-vlm  --dense-run "$DENSE" --split test --backend "${BACKEND:-openai}" --run-name B4_test
instrument-ir rerank-agent --dense-run "$DENSE" --split test --backend "${BACKEND:-openai}" --ablation full --run-name B5_test
for r in B4_test B5_test; do instrument-ir evaluate --run "outputs/runs/${r}.trec" --qrels data/processed/qrels/test.qrels; done
instrument-ir report
