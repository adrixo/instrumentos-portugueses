#!/usr/bin/env bash
# FASE 1 — Smoke en GPU: verifica que TODO funciona end-to-end (B1, B3, B4, B5) sobre un subset
# pequeño antes de lanzar el run completo. Pensado para la caja CUDA (RunPod/Lambda/Vast).
#
# Requisitos previos (ver DEPLOY_GPU.md):
#   - paquete instalado:  pip install -e ".[dense,colpali,rerank,extras]"
#   - dataset en data/raw/portuguese_instruments
#   - servidor vLLM sirviendo Qwen2.5-VL en $VLM_BASE_URL (modelo $VLM_MODEL)
set -euo pipefail
trap 'rc=$?; [ $rc -ne 0 ] && bash scripts/notify.sh "❌ SMOKE falló (rc=$rc)"' EXIT

VLM_BASE_URL="${VLM_BASE_URL:-http://localhost:8001/v1}"
VLM_MODEL="${VLM_MODEL:-qwen2.5-vl}"
QM=configs/queries_mini.yaml
QR=data/processed/qrels/mini.qrels
SMOKE_INSTR="${SMOKE_INSTR:-adufe,concertina}"
SMOKE_N="${SMOKE_N:-40}"
TOPN="${SMOKE_TOPN:-15}"

echo "### 0. prepare-data + subset mini ($SMOKE_INSTR, $SMOKE_N imgs, top_n=$TOPN)"
instrument-ir prepare-data
instrument-ir prepare-mini --instruments-sel "$SMOKE_INSTR" --n-images "$SMOKE_N"

echo "### 1. B1 dense (OpenCLIP)"
instrument-ir retrieve --split mini --model openclip-vitb32 --top-k $TOPN --queries $QM --run-name B1_smoke
instrument-ir evaluate --run outputs/runs/B1_smoke.trec --qrels $QR --queries $QM
bash scripts/save_results.sh "smoke B1"

echo "### 2. B3 late-interaction (ColQwen) — VERIFICACIÓN CLAVE (falló en Mac)"
instrument-ir retrieve --split mini --model colqwen --top-k $TOPN --queries $QM --run-name B3_smoke
instrument-ir evaluate --run outputs/runs/B3_smoke.trec --qrels $QR --queries $QM
bash scripts/save_results.sh "smoke B3 (ColQwen)"

echo "### 3. B4 VLM reranker (vía vLLM)"
instrument-ir rerank-vlm --dense-run outputs/runs/B1_smoke.trec --split mini \
  --backend openai --base-url "$VLM_BASE_URL" --vlm-model "$VLM_MODEL" \
  --top-n $TOPN --final-top-k $TOPN --queries $QM --run-name B4_smoke
instrument-ir evaluate --run outputs/runs/B4_smoke.trec --qrels $QR --queries $QM
bash scripts/save_results.sh "smoke B4"

echo "### 4. B5 agente (vía vLLM)"
instrument-ir rerank-agent --dense-run outputs/runs/B1_smoke.trec --split mini \
  --backend openai --base-url "$VLM_BASE_URL" --vlm-model "$VLM_MODEL" \
  --ablation full --top-n $TOPN --final-top-k $TOPN --queries $QM --run-name B5_smoke
instrument-ir evaluate --run outputs/runs/B5_smoke.trec --qrels $QR --queries $QM
bash scripts/save_results.sh "smoke B5"

echo "### 5. report"
instrument-ir report
bash scripts/save_results.sh "smoke report"

echo ""
echo "===================================================================="
echo " SMOKE GPU OK — B1/B3/B4/B5 ejecutados de extremo a extremo."
echo " Revisa outputs/reports/final_report.md y outputs/metrics/*_smoke.json"
echo " Si B3 dio métricas razonables (no aleatorias), lanza scripts/gpu_full.sh"
echo "===================================================================="
bash scripts/notify.sh "✅ SMOKE GPU terminado — revisa resultados en GitHub y lanza el gordo"

# Auto-apagado opcional también en el smoke (para probar el apagado). Default: NO apaga.
if [ "${SHUTDOWN:-0}" = "1" ]; then
  bash scripts/notify.sh "⏻ (smoke) Apagando el Pod en ${SHUTDOWN_GRACE:-60}s..."
  bash scripts/shutdown_pod.sh
fi
