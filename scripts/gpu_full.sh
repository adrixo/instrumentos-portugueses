#!/usr/bin/env bash
# FASE 2 — Run completo en GPU (cifras del paper). Lánzalo SOLO si gpu_smoke.sh fue bien.
# B1 (3 dense) + B3 (ColQwen) en valid+test; B4 y B5 (+ablaciones) en test con top_n=200.
set -euo pipefail
trap 'rc=$?; [ $rc -ne 0 ] && bash scripts/notify.sh "❌ RUN COMPLETO falló (rc=$rc) — revisa logs en el Pod"' EXIT

VLM_BASE_URL="${VLM_BASE_URL:-http://localhost:8001/v1}"
VLM_MODEL="${VLM_MODEL:-qwen2.5-vl}"
TOPN="${TOPN:-200}"
FINAL_K="${FINAL_K:-100}"
DENSE_MODEL="${DENSE_MODEL:-openclip-vitl14}"   # dense base para B4/B5 (mejor validado)

echo "### prepare-data + qrels"
instrument-ir prepare-data
for s in valid test; do instrument-ir build-qrels --split "$s"; done

echo "### B1 dense (3 modelos) en valid+test"
for s in valid test; do
  for m in openclip-vitb32 openclip-vitl14 jinaclip; do
    instrument-ir retrieve --split "$s" --model "$m" --top-k 100 --run-name "B1_${m}_${s}"
    instrument-ir evaluate --run "outputs/runs/B1_${m}_${s}.trec" --qrels "data/processed/qrels/${s}.qrels"
    bash scripts/save_results.sh "full B1 ${m} ${s}"
  done
done

echo "### B3 ColQwen en valid+test"
for s in valid test; do
  instrument-ir retrieve --split "$s" --model colqwen --top-k 100 --run-name "B3_colqwen_${s}"
  instrument-ir evaluate --run "outputs/runs/B3_colqwen_${s}.trec" --qrels "data/processed/qrels/${s}.qrels"
  bash scripts/save_results.sh "full B3 colqwen ${s}"
done

# Dense base para reranking (top_n) en test. Re-recuperamos top_n con el mejor dense.
DENSE_RUN="outputs/runs/B1_${DENSE_MODEL}_test.trec"
echo "### candidatos top-$TOPN para reranking (dense=$DENSE_MODEL, test)"
instrument-ir retrieve --split test --model "$DENSE_MODEL" --top-k "$TOPN" --run-name "DENSE_test"
DENSE_RUN="outputs/runs/DENSE_test.trec"

echo "### B4 VLM reranker (test, top_n=$TOPN)"
instrument-ir rerank-vlm --dense-run "$DENSE_RUN" --split test --backend openai \
  --base-url "$VLM_BASE_URL" --vlm-model "$VLM_MODEL" --top-n "$TOPN" --final-top-k "$FINAL_K" \
  --run-name "B4_test"
instrument-ir evaluate --run outputs/runs/B4_test.trec --qrels data/processed/qrels/test.qrels
instrument-ir rerank-metrics --dense-run "$DENSE_RUN" --reranked-run outputs/runs/B4_test.trec \
  --qrels data/processed/qrels/test.qrels --n "$TOPN" --k "$FINAL_K"
bash scripts/save_results.sh "full B4 test"

echo "### B5 agente + ablaciones (test, top_n=$TOPN)"
for abl in full no_crops no_caption full_image_only max_score_only weighted_fusion; do
  instrument-ir rerank-agent --dense-run "$DENSE_RUN" --split test --backend openai \
    --base-url "$VLM_BASE_URL" --vlm-model "$VLM_MODEL" --ablation "$abl" \
    --top-n "$TOPN" --final-top-k "$FINAL_K" --run-name "B5_${abl}_test"
  instrument-ir evaluate --run "outputs/runs/B5_${abl}_test.trec" --qrels data/processed/qrels/test.qrels
  bash scripts/save_results.sh "full B5 ${abl} test"
done
instrument-ir rerank-metrics --dense-run "$DENSE_RUN" --reranked-run outputs/runs/B5_full_test.trec \
  --qrels data/processed/qrels/test.qrels --n "$TOPN" --k "$FINAL_K" \
  --out outputs/metrics/B5_full_test__rerankmetrics.json

echo "### error-analysis + report"
instrument-ir error-analysis --run outputs/runs/B4_test.trec --qrels data/processed/qrels/test.qrels --out outputs/reports/error_B4_test.json
instrument-ir error-analysis --run outputs/runs/B5_full_test.trec --qrels data/processed/qrels/test.qrels --out outputs/reports/error_B5_test.json
instrument-ir report
bash scripts/save_results.sh "full report (FINAL)"

echo ""
echo "===================================================================="
echo " RUN COMPLETO OK. Resultados en outputs/reports/final_report.md"
echo " Tablas: outputs/reports/tables/  (macro, per-class, gain con p-value)"
echo "===================================================================="
bash scripts/notify.sh "✅ RUN COMPLETO (gordo) terminado — resultados en GitHub"

# Auto-apagado del Pod si SHUTDOWN=1 (los resultados ya están en GitHub).
if [ "${SHUTDOWN:-0}" = "1" ]; then
  bash scripts/notify.sh "⏻ Apagando el Pod en ${SHUTDOWN_GRACE:-60}s..."
  bash scripts/shutdown_pod.sh
fi
