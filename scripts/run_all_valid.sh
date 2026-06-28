#!/usr/bin/env bash
# Pipeline de desarrollo en valid (ADR §17.1).
set -euo pipefail
instrument-ir prepare-data
instrument-ir build-qrels --split valid
for m in openclip-vitb32 openclip-vitl14 jinaclip; do
  instrument-ir retrieve --split valid --model "$m" --run-name "B1_${m}_valid"
  instrument-ir evaluate --run "outputs/runs/B1_${m}_valid.trec" --qrels data/processed/qrels/valid.qrels
done
