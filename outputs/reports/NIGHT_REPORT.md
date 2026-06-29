# Night Report — Instrument Retrieval Lab (simulación de artículo)

**Fecha:** 2026-06-29 (run nocturno, ~00:00–03:10) · **Máquina:** Apple M4 Max (MPS), 36 GB
**Objetivo del run:** validar de punta a punta, con **modelos reales**, todo el pipeline (B1 dense →
B4 VLM reranker → B5 agente determinista + ablaciones) sobre un subconjunto, y comprobar que los
datos son correctos, sin fuga y con sentido científico.

> ⚠️ Esto es una **simulación a pequeña escala** para validar el método, no los números finales del
> paper. Escala: 6 instrumentos, 120 imágenes, queries EN, top-N=30. Para conclusiones estadísticas
> hace falta el run completo (22 clases, valid+test) en GPU.

---

## 1. Setup

- **Corpus:** 120 imágenes del split `valid` (subset `mini`), positivos de 6 instrumentos:
  adufe, concertina, cavaquinho, gaita-de-foles, guitarra-portuguesa, violão (mezcla fácil/difícil).
- **Queries:** 6 (una por instrumento, EN). **Ground truth:** COCO (multi-etiqueta).
- **Sistemas:**
  - **B1** — dense global, OpenCLIP ViT-B/32.
  - **B4** — B1 top-30 → reranker VLM pointwise (Qwen2.5-VL-3B, JSON cerrado).
  - **B5_full** — B1 top-30 → agente determinista (VQA imagen completa → si duda: caption + crops
    center/grid2×2 + VQA de crops → fusión max).
  - **Ablaciones B5:** `no_crops`, `full_image_only`.
- **VLM:** Qwen2.5-VL-3B-Instruct in-process (MPS), greedy, temperature=0, seed=42.
- **Determinismo:** seed fijo; B5 traza estructurada por candidato (sin chain-of-thought libre).

---

## 2. Resultados principales (macro sobre 6 instrumentos)

| Sistema | nDCG@10 | mAP | MRR | Recall@10 | Recall@30 | P@10 |
|---|---|---|---|---|---|---|
| B1 (dense) | 0.557 | 0.361 | 0.715 | 0.210 | 0.529 | 0.533 |
| B4 (VLM rerank) | 0.603 | 0.396 | 0.867 | 0.244 | 0.529 | 0.550 |
| **B5_full (agente)** | **0.623** | **0.418** | **0.917** | **0.247** | 0.529 | 0.550 |
| B5_no_crops | 0.603 | 0.396 | 0.867 | 0.244 | 0.529 | 0.550 |
| B5_full_image_only | 0.603 | 0.396 | 0.867 | 0.244 | 0.529 | 0.550 |

**Métricas de reranking** (techo y ganancia):
- `candidate_recall@30 = oracle_recall@30 = 0.529` para todos → el reranker hereda el techo del dense.
- `rerank_gain@30 = 0.0` (esperado: reordenar los mismos 30 no cambia el recall@30).
- `delta_nDCG@30`: B4 **+0.042**, B5_full **+0.059**. `delta_mAP`: B4 **+0.034**, B5_full **+0.056**.

---

## 3. Lectura científica (¿tiene sentido?)

1. **El reranker mejora el ORDEN, no el recall.** Recall@30 es idéntico en B1/B4/B5 porque el reranker
   solo reordena los 30 candidatos del dense; su valor está en subir los relevantes (nDCG/mAP/MRR↑),
   no en encontrar nuevos. Es correcto y es justo el comportamiento esperado de un reranker de 2 etapas
   (su techo es `oracle_recall`, aquí 0.529).
2. **B5_full > B4 > B1** en todas las métricas de orden. **La parte agéntica (crops+caption) aporta**:
   sobre el VLM pointwise, B5_full añade +0.019 nDCG@10, +0.022 mAP, +0.050 MRR.
3. **Ablaciones limpias:** `no_crops` y `full_image_only` dan resultados **idénticos a B4** (a 4 decimales).
   Es lo correcto: sin crops, B5 ≡ VLM pointwise. Confirma determinismo y **aísla la ganancia a los crops**.
4. **Por instrumento (Recall@10, B1 → B5_full):**
   - guitarra-portuguesa **0.10 → 0.40** (gran mejora del agente),
   - adufe 0.03 → 0.07 (el más difícil, sube poco),
   - concertina 0.29 → 0.29 (ya fácil, sin cambio),
   - cavaquinho 0.21 → 0.12 y gaita 0.36 → 0.36 (**el agente puede empeorar** algún caso).
   → Patrón realista: el agente ayuda en instrumentos confundibles/pequeños pero no es uniformemente
   mejor. Material de "error analysis" para el paper.

5. **Coste (cost/quality):** B4 ≈ 13 s/candidato, B5_full ≈ 22 s/candidato (los crops ~+70 % de coste
   por +0.02 nDCG). B5_no_crops/full_image ≈ 13 s (≈ B4). Dato clave para la discusión coste-beneficio.

---

## 4. Significancia estadística (honesto)

Con **n=6 queries** no hay potencia: las diferencias son positivas pero **no significativas**
(test de permutación pareado, IC95% bootstrap):

| Comparación (nDCG@10) | Δ | IC95% | p |
|---|---|---|---|
| B4 − B1 | +0.046 | [−0.05, +0.18] | 0.69 |
| B5_full − B4 | +0.019 | [−0.09, +0.13] | 0.75 |
| B5_full − B1 | +0.066 | [−0.11, +0.26] | 0.63 |
| B5_full − B1 (MRR) | +0.201 | [−0.17, +0.58] | 0.50 |

→ **Tendencia correcta, sin significancia** (esperado a esta escala). La significancia exige el run
completo (22 clases × idiomas, valid+test) en GPU.

---

## 5. Integridad de datos / control de fuga (QA)

- **Traces** (B4, B5×3): 180 líneas cada uno (6×30). **Fuga = 0** (ningún nombre de archivo ni id de
  Vimeo), **todos los `image_id` anónimos**, y los de B5 **validan contra `agent_trace.schema.json`**.
- **Runfiles** TREC válidos; **métricas** en rango [0,1], sin NaN, coherentes (B4/B5 ≥ B1).
- **Anti-fuga estructural** intacto: el modelo solo vio píxeles + nombre del instrumento consultado.

---

## 6. Anomalías encontradas y resueltas esta noche

1. **B3 (ColQwen) incompatible** con `transformers 5.12`: el adaptador LoRA de `colqwen2-v1.0` usa la
   nomenclatura antigua (`model.*` vs `language_model.*`) → no se aplica (pesos LoRA aleatorios) y el
   proceso se colgaba. **Acción:** B3 omitido del run; documentado. Evidencia en
   `outputs/night/_b3_issue/`. **Pendiente:** ejecutarlo en el box GPU con `transformers` fijado.
2. **`rerank-metrics` diluía** candidate/oracle_recall promediando sobre las 66 queries del qrels en
   vez de las 6 evaluadas (daba 0.066). **Corregido** (intersección con el run) y **recalculado** → 0.529.
3. **Watchdog medía el proceso equivocado** (orquestador, 0 % CPU) → habría reiniciado el run cada
   10 min y B5 nunca habría terminado. **Corregido** (mide CPU de todo el grupo).

Ninguna anomalía invalida los resultados; las tres están arregladas.

---

## 7. Conclusiones

- **El pipeline completo funciona de extremo a extremo con modelos reales en esta máquina (MPS).**
- **El reranking ayuda al orden** (nDCG/mAP/MRR), y **el agente (B5_full) supera al VLM pointwise (B4)**,
  con la ganancia atribuible a los crops (ablaciones lo confirman).
- El efecto es **heterogéneo por instrumento** (gran ayuda en guitarra-portuguesa; nula/negativa en
  cavaquinho), lo que da una historia de error analysis interesante.
- A escala de 6 queries **no hay significancia**: el valor del run es metodológico (validación), no de
  conclusión estadística.

## 8. Próximos pasos para el artículo

1. **Run completo en GPU** (CUDA + vLLM): 22 clases × idiomas, valid+test, top-N=200 → tablas con
   significancia (bootstrap + permutación + Holm), que el pipeline ya genera (`make report`).
2. **Arreglar B3** (entorno con `transformers` fijado o `colqwen2.5`) e incluirlo en la comparativa.
3. **Error analysis** cualitativo (galerías FP/FN ya generadas: `outputs/reports/error_B*_mini.json`).
4. Opcional: VLM mayor (7B) en GPU para mejor calidad de reranking.

---

### Artefactos de este run
- Runs: `outputs/runs/{B1,B4,B5_full,B5_nocrops,B5_fullimg}_mini.trec`
- Métricas: `outputs/metrics/*_mini*.json` · Traces: `outputs/rerank_traces/*_mini.jsonl`
- Error analysis: `outputs/reports/error_{B4,B5}_mini.json` · Estado run: `outputs/night/summary.json`
- Informe auto + tablas/figuras: `outputs/reports/final_report.md`, `outputs/reports/tables/`
