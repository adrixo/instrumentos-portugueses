# QA del test nocturno — notas

**2026-06-29 ~06:15** · revisión automática tras `DONE`.

## Comprobaciones superadas
- **Anti-fuga**: ninguna traza (B4/B5_full/B5_nocrops/B5_fullimg) contiene nombres de archivo ni ids de Vimeo; todos los `image_id` son anónimos. ✅
- **Schema**: las trazas B5 validan contra `schemas/agent_trace.schema.json`. ✅
- **Formato**: runfiles TREC con 6 columnas; 180 líneas por traza = 6 queries × 30 candidatos. ✅
- **Pasos**: `summary.json` sin error/timeout; B4 ~40 min, B5_full ~66 min, ablaciones ~40 min c/u.
- **Rango**: métricas en [0,1], sin NaN.

## Anomalía encontrada y corregida
- **`candidate_recall@30` / `oracle_recall@30` salían 0.066** (deberían ser ~0.53). Causa: el macro se promediaba sobre **todas las 48 queries del qrels** en lugar de solo las **6 evaluadas** presentes en el run (3.172/48=0.066 vs 3.172/6=0.529). El cálculo manual de recall@30 confirmó 0.529 (coincide con `evaluate`).
- **Acción**: el bug ya estaba corregido en `rerank_metrics.py` (intersección `set(qrels) & set(run)`). Se **re-ejecutó** el paso `rerank-metrics` para B4 y B5 → valores correctos (candidate/oracle_recall@30 = **0.5287**). Los JSON de métricas de reranking quedan actualizados.

## Notas (no son errores)
- **`rerank_gain@30 = 0` es estructural**: en el mini se usó `top_n = final_top_k = 30`, así que el reranker solo **reordena** el mismo conjunto → recall@K no puede cambiar para K≥30. La mejora del reranker aparece en métricas de posición (nDCG@10, mAP, MRR), no en recall@30. En el run real (top_n=200 → final 100) el gain de recall sí será medible.
- **B3 (ColQwen) omitido**: adaptador LoRA de `colqwen2-v1.0` incompatible con transformers 5.12 (renombrado `language_model.*` → el LoRA no se aplica). Evidencia en `outputs/night/_b3_issue/`. Se resuelve fijando versión de transformers en el box GPU.
