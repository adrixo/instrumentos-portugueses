# PRD — Multimodal Dense Retrieval + VLM/Agentic Reranking para instrumentos tradicionales portugueses

Dataset previo: https://www.sciencedirect.com/science/article/pii/S2352340925004664
Los ids de las imagenes corresponden a videos de vimeo. Informate en el paper en si.

## 0. Rol del agente de código

Actúa como ingeniero senior de Machine Learning, MLOps y experimentación reproducible. Debes implementar un repositorio completo, ejecutable en una máquina remota con GPU mediante contenedores Docker, para evaluar sistemas de recuperación de imágenes de instrumentos tradicionales portugueses.

El objetivo no es entrenar un clasificador YOLO ni un detector supervisado, sino construir y evaluar un sistema de **Information Retrieval visual**:

> Dada una consulta textual como “encuéntrame imágenes con adufes”, recuperar y ordenar las imágenes del dataset que contienen ese instrumento.

No usar BM25 ni búsqueda textual sobre metadatos como baseline principal, porque el corpus experimental se tratará como puramente visual.

---

# 1. Objetivo general

Construir un framework experimental reproducible para comparar:

* **B1 — Dense multimodal retrieval global**
* **B3 — Late-interaction visual retrieval**
* **B4 — Dense retrieval + VLM reranker**
* **B5 — Dense retrieval + deterministic agentic multimodal reranker**

El resultado debe incluir:

1. Repositorio de código limpio.
2. Contenedores Docker con GPU.
3. Pipelines reproducibles.
4. Evaluación IR por instrumento.
5. Trazas estructuradas de los sistemas con VLM/agente.
6. Herramienta final de búsqueda visual.
7. Informes automáticos.
8. Presentación final en Markdown usando Slidev, Marp o Reveal.js.

---

# 2. Contexto científico

El dataset contiene imágenes de instrumentos tradicionales portugueses extraídas de vídeos y anotadas para visión por computador. Cada imagen puede contener uno o varios instrumentos.

Este proyecto transforma el problema desde “clasificación/detección supervisada” hacia “búsqueda visual por consulta textual”, lo que permite formular preguntas como:

```text
Find records containing an adufe.
Find images with a Portuguese guitar.
Find records where a concertina appears.
```

El ground truth se construirá a partir de las anotaciones COCO existentes:

* Una imagen es relevante para una query de clase `C` si contiene al menos una anotación con clase `C`.
* Una imagen puede ser relevante para varias queries si contiene varios instrumentos.
* No se deben usar nombres de archivo, rutas ni metadatos con etiquetas como señal de ranking.

---

# 3. Hipótesis de investigación

## H1 — Dense retrieval global

Los modelos multimodales globales tipo CLIP/OpenCLIP/JinaCLIP/BGE-VL/Qwen3-VL-Embedding permiten recuperar imágenes relevantes sin entrenamiento específico, pero fallan en instrumentos pequeños, ocluidos o visualmente parecidos.

## H2 — Late interaction visual retrieval

Los modelos tipo ColPali/ColQwen, al usar representaciones multivectoriales visuales y late interaction, mejoran la recuperación frente a embeddings globales, especialmente cuando el instrumento ocupa una zona pequeña de la imagen.

## H3 — VLM reranking

Un VLM usado como reranker sobre el top-N recuperado por dense retrieval mejora el ranking final porque inspecciona explícitamente la imagen y decide si el instrumento está presente.

## H4 — Agentic reranking

Un workflow agentic controlado, con herramientas locales como captioning, VQA, crops y verificación visual, mejora el reranking frente a una única llamada VLM sobre imagen completa, especialmente en casos ambiguos.

---

# 4. Sistemas experimentales

## B1 — Dense multimodal retrieval global

### Descripción

Sistema base de recuperación multimodal. Cada imagen se codifica en un vector global y cada query textual se codifica en el mismo espacio. Se ordenan las imágenes por similitud coseno o producto punto.

### Modelos mínimos

Implementar al menos:

```text
open_clip ViT-B-32
open_clip ViT-L-14
jina-clip-v2
BGE-VL / Visualized BGE si es viable
Qwen3-VL-Embedding si el entorno lo permite
```

Comprobrar cual es el SOTA multimodal en MTEB.

### Entrada

```text
query: "adufe"
image corpus: imágenes del split configurado
```

### Salida

Runfile por modelo:

```text
query_id Q0 image_id rank score run_name
```

### Objetivo

Establecer baseline fuerte y barato.

---

## B3 — Late-interaction visual retrieval

### Descripción

Sistema de recuperación visual con embeddings multivectoriales. Cada imagen no se representa con un único vector, sino con múltiples vectores asociados a patches/tokens visuales. La query textual se compara con esos vectores mediante late interaction.

### Modelos candidatos

```text
ColPali
ColQwen2
ColQwen2.5
ColQwen3-style retriever si está disponible
```

### Pipeline

```text
1. Cargar imágenes.
2. Calcular embeddings multivectoriales.
3. Guardar índice.
4. Codificar queries.
5. Calcular scores query-imagen.
6. Recuperar top-K.
7. Exportar runfiles.
8. Evaluar con qrels.
```

### Salida esperada

```text
outputs/runs/B3_colqwen2_test.trec
outputs/metrics/B3_colqwen2_test.json
outputs/artifacts/B3_colqwen2_index_stats.json
```

### Métricas principales

```text
Recall@20
Recall@50
Recall@100
nDCG@20
nDCG@100
mAP
MRR
latency_ms_per_query
index_size_mb
```

---

## B4 — Dense retrieval + VLM pointwise reranker

### Descripción

Sistema de dos etapas:

```text
Primera etapa:
    dense retrieval recupera top-N candidatos.

Segunda etapa:
    un VLM evalúa cada candidato individualmente y asigna un score de relevancia.
```

### Flujo

```text
query -> dense retriever -> top-200 candidates -> VLM reranker -> final top-100
```

### Dense retriever base

Debe ser configurable:

```text
B1_best_model
B3_best_model
```

Por defecto, usar el mejor modelo validado en B1/B3 sobre el split de validación.

### Reranker VLM

Debe soportar backends intercambiables:

```text
qwen2.5-vl-instruct
qwen3-vl-reranker si está disponible
internvl si está disponible
llava-next si está disponible
API local OpenAI-compatible si se configura
```

El backend debe ejecutarse localmente o contra un endpoint controlado de la máquina remota. No se permite internet search.

### Prompt cerrado

Usar salida JSON estricta.

```text
You are evaluating whether an image contains a target traditional Portuguese musical instrument.

Target instrument: {instrument_name}
Target definition: {instrument_definition}

Rules:
- Use only visible evidence in the image.
- Do not infer from filename, path, dataset split or metadata.
- If unsure, choose "uncertain".
- Return only valid JSON.

JSON schema:
{
  "decision": "present" | "absent" | "uncertain",
  "confidence": 0.0,
  "visual_evidence": ["short visible evidence"],
  "negative_evidence": ["short uncertainty or absence evidence"],
  "score": 0
}
```

### Score

Normalizar `score` a `[0, 1]`.

Regla por defecto:

```text
if decision == "present":
    final_score = confidence
elif decision == "uncertain":
    final_score = 0.5 * confidence
else:
    final_score = 0.0
```

Tie-breaker:

```text
dense_score
```

### Artefactos obligatorios

```text
outputs/candidates/{run_id}.parquet
outputs/rerank_traces/{run_id}.jsonl
outputs/runs/{run_id}.trec
outputs/metrics/{run_id}.json
```

Cada línea de trace:

```json
{
  "run_id": "B4_colqwen2_qwen25vl_test_seed42",
  "query_id": "q_adufe_en",
  "instrument": "adufe",
  "image_id": "anon_000001",
  "dense_rank": 87,
  "dense_score": 0.312,
  "vlm_decision": "present",
  "vlm_confidence": 0.86,
  "vlm_score": 0.86,
  "final_score": 0.86,
  "visual_evidence": [
    "square frame drum visible",
    "held by performer"
  ],
  "negative_evidence": [],
  "model": "qwen2.5-vl-7b-instruct",
  "temperature": 0.0,
  "seed": 42,
  "timestamp_utc": "..."
}
```

---

## B5 — Dense retrieval + deterministic agentic multimodal reranker

### Descripción

Sistema de reranking agentic controlado. No debe ser un agente libre que decida cualquier cosa. Debe ser un workflow determinista con herramientas locales.

### Diferencia con B4

B4:

```text
imagen completa + pregunta -> score
```

B5:

```text
imagen completa -> caption
imagen completa -> VQA directa
si duda -> crops
crops -> VQA local
opcional -> metadata permitida
evidencias -> score final
```

### Herramientas locales del agente

Implementar como funciones puras siempre que sea posible.

```text
retrieve_candidates(query, top_k)
caption_image(image_id)
ask_vlm_full_image(image_id, instrument)
generate_crops(image_id)
ask_vlm_crop(crop_id, instrument)
get_safe_metadata(image_id)
score_evidence(evidence)
write_trace(trace)
```

### Herramientas prohibidas

```text
internet_search
web_browser
reading filenames containing labels
reading folder names containing labels
reading COCO annotation of candidate during inference
using ground truth during reranking
using train labels to construct prompts for test images
```

### Metadata permitida

Por defecto, ninguna metadata se pasa al modelo.

Solo se permitirá metadata si está explícitamente marcada como segura y no contiene etiquetas de instrumentos. El modo por defecto debe ser:

```yaml
use_metadata: false
```

### Grafo de decisión

Implementar con LangGraph o con un grafo propio sencillo. Preferido: LangGraph.

Pseudoflujo:

```text
START
  -> full_image_vqa
  -> if confidence >= HIGH_CONF:
         final_score
     else:
         caption_image
         generate_crops
         crop_vqa
         evidence_fusion
  -> final_ranking
END
```

Parámetros por defecto:

```yaml
high_confidence_threshold: 0.80
low_confidence_threshold: 0.40
max_crops_per_image: 5
top_n_candidates: 200
final_top_k: 100
temperature: 0.0
seed: 42
```

### Crops

Implementar tres estrategias, activables por configuración:

```text
center_crop
grid_crops_2x2
saliency_or_objectness_crops
```

La primera versión puede implementar solo:

```text
full_image
center_crop
grid_2x2
```

Más adelante se puede añadir GroundingDINO/SAM, pero no es obligatorio para el MVP.

### Evidence fusion

Primera versión simple:

```text
score_full = VLM full-image score
score_crop = max crop VLM score
score_caption = caption match score

final_score = max(score_full, score_crop)
tie_breaker = dense_score
```

Versión opcional:

```text
final_score =
    0.60 * max(score_full, score_crop)
  + 0.20 * score_caption
  + 0.20 * dense_score_normalized
```

Para el paper, priorizar la versión simple porque es más defendible.

### Trace B5

Cada candidato debe guardar una traza estructurada:

```json
{
  "run_id": "B5_colqwen2_agent_qwen25vl_test_seed42",
  "query_id": "q_adufe_en",
  "instrument": "adufe",
  "image_id": "anon_000001",
  "dense_rank": 87,
  "dense_score": 0.312,
  "steps": [
    {
      "tool": "ask_vlm_full_image",
      "decision": "uncertain",
      "confidence": 0.55,
      "score": 0.275,
      "evidence": ["possible square percussion object"]
    },
    {
      "tool": "caption_image",
      "caption": "A group of musicians performing; one person may be holding a square percussion instrument."
    },
    {
      "tool": "generate_crops",
      "crops": ["crop_01", "crop_02", "crop_03", "crop_04"]
    },
    {
      "tool": "ask_vlm_crop",
      "crop_id": "crop_02",
      "decision": "present",
      "confidence": 0.84,
      "score": 0.84,
      "evidence": ["square frame drum visible in hands"]
    }
  ],
  "final_decision": "present",
  "final_score": 0.84,
  "final_rank": 12,
  "seed": 42,
  "temperature": 0.0,
  "model_versions": {
    "dense_retriever": "colqwen2-v1.0",
    "vlm": "qwen2.5-vl-7b-instruct",
    "agent_framework": "langgraph"
  }
}
```

No guardar chain-of-thought libre. Guardar solo decisiones, scores y evidencias visuales breves.

---

# 5. Dataset y ground truth

## 5.1 Entrada esperada

El agente debe asumir que el usuario colocará el dataset descargado en:

```text
data/raw/portuguese_instruments/
```

Estructura esperada aproximada:

```text
data/raw/portuguese_instruments/
  train/
    images/
    _annotations.coco.json
  valid/
    images/
    _annotations.coco.json
  test/
    images/
    _annotations.coco.json
```

El código debe detectar automáticamente variantes comunes:

```text
valid/
val/
validation/
_annotations.coco.json
instances_*.json
```

## 5.2 Anonimización obligatoria

El dataset puede contener etiquetas en nombres de archivo. Por tanto:

* Crear `image_id` interno anónimo.
* No pasar nombres de archivo al modelo.
* No pasar rutas al modelo.
* No guardar nombres originales en traces públicos.
* Mantener mapping privado:

```text
data/processed/image_id_mapping.parquet
```

Este mapping solo se usa para cargar imágenes, nunca para inferencia.

## 5.3 Construcción de qrels

Para cada split:

```text
qrels/{split}.qrels
```

Formato TREC:

```text
query_id 0 image_id relevance
```

Donde:

```text
relevance = 1 si la imagen contiene la clase de la query
relevance = 0 no se escribe normalmente en qrels
```

## 5.4 Queries

Generar automáticamente queries por clase.

Para cada clase:

```yaml
instrument_id: adufe
queries:
  - id: q_adufe_pt
    text: "adufe"
    language: "pt"
  - id: q_adufe_en
    text: "Portuguese square frame drum adufe"
    language: "en"
  - id: q_adufe_es
    text: "tambor cuadrado portugués adufe"
    language: "es"
```

Debe existir un archivo editable:

```text
configs/queries.yaml
```

El estudiante podrá mejorar definiciones y sinónimos.

---

# 6. Evaluación

## 6.1 Métricas IR

Calcular por query y agregadas:

```text
Recall@10
Recall@20
Recall@50
Recall@100
Precision@10
Precision@20
Precision@50
Precision@100
nDCG@10
nDCG@20
nDCG@100
AP
mAP
MRR
R-Precision
```

## 6.2 Métricas de reranking

Para B4 y B5, calcular también:

```text
candidate_recall@200
oracle_recall@200
rerank_gain@100
delta_nDCG@100
delta_mAP
```

`oracle_recall@200` es crítico: mide cuántas imágenes relevantes estaban ya dentro del top-200 inicial. Si no estaban ahí, el reranker no podía recuperarlas.

## 6.3 Métricas de coste

Registrar:

```text
index_build_time_seconds
embedding_time_seconds
query_latency_ms
rerank_latency_ms_per_candidate
total_rerank_time_seconds
gpu_memory_peak_mb
index_size_mb
num_vlm_calls
num_crop_calls
```

## 6.4 Análisis estadístico

Usar `ranx` o `ir_measures`.

Implementar:

```text
bootstrap 95% CI
paired tests entre sistemas
tabla LaTeX
tabla Markdown
```

Comparaciones mínimas:

```text
B1_best vs B3_best
B3_best vs B4
B4 vs B5
```

---

# 7. Modos de evaluación

## 7.1 Modo desarrollo

Usar `valid` para:

* Elegir modelo dense base.
* Ajustar prompts.
* Ajustar umbrales del agente.
* Elegir top-N de candidatos.

## 7.2 Modo final

Usar `test` para:

* Métricas finales.
* Tablas finales.
* Figuras finales.
* Error analysis.

No ajustar prompts ni umbrales después de mirar resultados de test.

## 7.3 Modo catálogo completo

Opcional:

```text
split: all
```

Sirve para demo operacional, no para conclusiones principales del paper.

---

# 8. Reproducibilidad

## 8.1 Seeds

Fijar:

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Configurar:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Cuando algún modelo no sea totalmente determinista, documentarlo en el run.

## 8.2 Configuración

Usar Hydra, OmegaConf o Pydantic Settings.

Ejemplo:

```yaml
experiment:
  name: B4_colqwen2_qwen25vl_test
  seed: 42
  split: test

dataset:
  root: data/processed/portuguese_instruments
  qrels: data/processed/qrels/test.qrels
  anonymize_filenames: true

retriever:
  type: colqwen
  model_name: vidore/colqwen2-v1.0
  top_k: 200

reranker:
  type: vlm_pointwise
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
  temperature: 0.0
  max_new_tokens: 512
  output_schema: schemas/vlm_rerank.schema.json

evaluation:
  final_top_k: 100
  metrics:
    - recall@20
    - recall@50
    - recall@100
    - ndcg@20
    - ndcg@100
    - map
    - mrr

tracking:
  mlflow: true
  experiment_name: portuguese_instruments_ir
```

## 8.3 Versionado

Guardar en cada run:

```text
git_commit
docker_image_digest
dataset_doi
dataset_version
model_name
model_revision
python_version
cuda_version
gpu_name
seed
config_hash
```

## 8.4 DVC

Implementar `dvc.yaml` con stages:

```text
prepare_data
build_qrels
embed_images
build_index
retrieve
rerank_b4
rerank_b5
evaluate
report
```

---

# 9. Arquitectura del repositorio

Crear este repositorio:

```text
instrument-ir-agentic/
  README.md
  LICENSE
  pyproject.toml
  Makefile
  docker/
    Dockerfile.gpu
    docker-compose.gpu.yml
    entrypoint.sh
  configs/
    dataset.yaml
    queries.yaml
    models/
      b1_openclip.yaml
      b3_colqwen.yaml
      b4_vlm_rerank.yaml
      b5_agentic.yaml
    experiments/
      exp_b1_valid.yaml
      exp_b3_valid.yaml
      exp_b4_test.yaml
      exp_b5_test.yaml
  data/
    raw/
    processed/
  src/
    instrument_ir/
      __init__.py
      cli.py
      data/
        prepare_dataset.py
        coco_parser.py
        anonymize.py
        qrels.py
      retrieval/
        base.py
        openclip_retriever.py
        jina_clip_retriever.py
        bge_vl_retriever.py
        qwen_vl_embedding_retriever.py
        colpali_retriever.py
        faiss_index.py
        qdrant_index.py
      reranking/
        vlm_pointwise.py
        schemas.py
        prompts.py
      agent/
        graph.py
        tools.py
        scoring.py
        crop_generation.py
      evaluation/
        metrics.py
        ranx_eval.py
        statistical_tests.py
        error_analysis.py
      reporting/
        tables.py
        plots.py
        report_generator.py
        slides_generator.py
      serving/
        app.py
        api.py
        frontend.py
      utils/
        logging.py
        gpu.py
        reproducibility.py
        io.py
  schemas/
    vlm_rerank.schema.json
    agent_trace.schema.json
  scripts/
    check_gpu.sh
    download_models.sh
    run_all_valid.sh
    run_all_test.sh
  notebooks/
    01_dataset_exploration.ipynb
    02_error_analysis.ipynb
  outputs/
    runs/
    metrics/
    reports/
    traces/
    figures/
    slides/
  tests/
    test_coco_parser.py
    test_qrels.py
    test_no_filename_leakage.py
    test_runfile_format.py
    test_trace_schema.py
```

---

# 10. CLI obligatoria

Implementar CLI con Typer o Click.

Comandos mínimos:

```bash
instrument-ir prepare-data --raw data/raw/portuguese_instruments --out data/processed
instrument-ir build-qrels --split test
instrument-ir embed --config configs/models/b1_openclip.yaml --split test
instrument-ir build-index --config configs/models/b1_openclip.yaml --split test
instrument-ir retrieve --config configs/experiments/exp_b1_test.yaml
instrument-ir rerank-vlm --config configs/experiments/exp_b4_test.yaml
instrument-ir rerank-agent --config configs/experiments/exp_b5_test.yaml
instrument-ir evaluate --run outputs/runs/B4.trec --qrels data/processed/qrels/test.qrels
instrument-ir report --experiment portuguese_instruments_ir
instrument-ir serve --config configs/serving.yaml
```

También crear comandos Make:

```bash
make build
make check-gpu
make prepare
make smoke
make b1
make b3
make b4
make b5
make eval
make report
make serve
make slides
make repro
```

---

# 11. Docker y ejecución remota con GPU

## 11.1 Dockerfile

Crear `docker/Dockerfile.gpu` basado en imagen CUDA/PyTorch.

Debe incluir:

```text
Python 3.11 o 3.12
PyTorch CUDA
Transformers
Accelerate
OpenCLIP
FAISS GPU o FAISS CPU fallback
Qdrant client
Pillow
OpenCV
pandas
pyarrow
numpy
scikit-learn
ranx
ir_measures
mlflow
dvc
typer
fastapi
uvicorn
streamlit o gradio
```

## 11.2 docker-compose

Servicios mínimos:

```text
irlab
mlflow
qdrant opcional
```

Servicios opcionales:

```text
minio
jupyter
vlm-server
```

Ejemplo conceptual:

```yaml
services:
  irlab:
    build:
      context: ..
      dockerfile: docker/Dockerfile.gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_HOME=/models/huggingface
      - TRANSFORMERS_CACHE=/models/huggingface
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ../data:/workspace/data
      - ../outputs:/workspace/outputs
      - ../configs:/workspace/configs
      - ../models:/models
    ports:
      - "7860:7860"
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns
    ports:
      - "5000:5000"
    volumes:
      - ../outputs/mlruns:/mlruns

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ../outputs/qdrant:/qdrant/storage
```

## 11.3 Comprobación GPU

`scripts/check_gpu.sh` debe verificar:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

# 12. Herramienta final

Nombre propuesto:

```text
Instrument Retrieval Lab
```

## 12.1 Objetivo

Permitir a un usuario buscar visualmente instrumentos en el dataset y comparar rankings.

## 12.2 Interfaz

Puede implementarse en Gradio, Streamlit o FastAPI + frontend simple.

MVP recomendado: Gradio.

Pantalla principal:

```text
- Selector de instrumento/query
- Selector de sistema:
    B1 dense
    B3 ColQwen
    B4 VLM reranker
    B5 agentic reranker
- Campo de query libre
- Top-K
- Botón Search
- Galería de resultados
- Score
- Rank
- Evidencia visual si existe
- Botón para ver trace JSON
```

## 12.3 Pantalla de evaluación

```text
- Tabla de métricas por sistema
- Tabla de métricas por instrumento
- Gráfica Recall@K
- Gráfica nDCG@K
- Matriz sistema vs instrumento
- Galería de falsos positivos
- Galería de falsos negativos
```

## 12.4 Exportación

Debe exportar:

```text
CSV
JSON
Markdown report
LaTeX table
Slide deck Markdown
```

---

# 13. Reportes automáticos

Generar:

```text
outputs/reports/final_report.md
outputs/reports/final_report.html
outputs/reports/final_report.pdf opcional
outputs/reports/tables/results_macro.md
outputs/reports/tables/results_per_class.md
outputs/reports/figures/recall_at_k.png
outputs/reports/figures/ndcg_at_k.png
outputs/reports/figures/latency_vs_quality.png
```

## 13.1 Secciones del informe

```text
1. Objective
2. Dataset
3. Leakage prevention
4. Methods
5. Experimental setup
6. Results
7. Statistical significance
8. Error analysis
9. Qualitative examples
10. Limitations
11. Conclusions
```

---

# 14. Presentación final

Generar una carpeta:

```text
outputs/slides/
```

Con uno de estos formatos:

```text
Slidev preferido
Marp aceptado
Reveal.js aceptado
```

## 14.1 Recomendación

Usar Slidev si se quieren incluir:

```text
- código
- figuras
- tablas
- componentes web
- resultados reproducibles versionados con Git
```

Usar Marp si se quiere algo más simple y portable.

Usar Reveal.js si se quiere una presentación HTML más personalizable.

## 14.2 Estructura de slides

```text
1. Title
2. Motivation
3. Dataset
4. Problem formulation as visual IR
5. Ground truth construction
6. Leakage control
7. Baselines
8. B3: late-interaction retrieval
9. B4: VLM reranking
10. B5: agentic reranking
11. Experimental setup
12. Metrics
13. Main results
14. Per-instrument results
15. Error analysis
16. Qualitative examples
17. Cost/latency analysis
18. Conclusions
19. Future work
```

---

# 15. Prevención de fuga de información

Implementar tests automáticos.

## 15.1 Prohibido

```text
- Usar nombres de archivo como texto.
- Usar rutas de carpetas como texto.
- Usar labels COCO durante inferencia.
- Pasar ground truth al VLM.
- Usar internet durante inferencia.
- Usar metadata no revisada.
```

## 15.2 Test obligatorio

`test_no_filename_leakage.py`

Debe comprobar:

```text
- Los prompts no contienen filename original.
- Los prompts no contienen path original.
- Los prompts no contienen clase ground truth salvo el instrumento consultado.
- Los traces públicos solo contienen image_id anónimo.
```

## 15.3 Modo offline

Añadir variable:

```bash
EXPERIMENT_OFFLINE=true
```

Cuando está activa:

```text
- bloquear llamadas HTTP externas salvo endpoints locales permitidos
- bloquear herramientas web
- registrar cualquier intento de conexión
```

---

# 16. Prompts y definiciones de instrumentos

Crear archivo:

```text
configs/instruments.yaml
```

Ejemplo:

```yaml
adufe:
  canonical_name: "adufe"
  definitions:
    en: "A traditional Portuguese square frame drum, usually held by hand."
    es: "Tambor cuadrado tradicional portugués, normalmente sostenido con las manos."
    pt: "Tambor quadrado tradicional português, normalmente segurado com as mãos."
  visual_cues:
    - "square frame"
    - "flat percussion surface"
    - "held by performer"
    - "traditional folk context"
  confusing_classes:
    - "pandeiro"
    - "tambourine"
    - "drum"
```

El agente debe usar definiciones y pistas visuales, no etiquetas del ground truth de cada imagen.

---

# 17. Experimentos mínimos obligatorios

## 17.1 Validación

```text
EXP01_B1_openclip_valid
EXP02_B1_jinaclip_valid
EXP03_B3_colqwen_valid
EXP04_B4_bestdense_qwen25vl_valid
EXP05_B5_bestdense_agent_valid
```

## 17.2 Test final

```text
EXP10_B1_best_test
EXP11_B3_best_test
EXP12_B4_bestdense_qwen25vl_test
EXP13_B5_bestdense_agent_test
```

## 17.3 Ablaciones B5

```text
B5_full
B5_no_crops
B5_no_caption
B5_full_image_only
B5_max_score_only
B5_weighted_fusion
```

---

# 18. Formato de resultados

## 18.1 Tabla macro

```text
system | Recall@20 | Recall@50 | Recall@100 | nDCG@20 | nDCG@100 | mAP | MRR | latency/query | rerank cost
```

## 18.2 Tabla por instrumento

```text
instrument | positives | B1_R@100 | B3_R@100 | B4_R@100 | B5_R@100 | best_system
```

## 18.3 Tabla de ganancia

```text
comparison | delta_R@100 | delta_nDCG@100 | p_value | significant
```

---

# 19. Criterios de aceptación

El proyecto se considera completo cuando:

1. `make build` construye la imagen Docker.
2. `make check-gpu` detecta CUDA.
3. `make prepare` procesa el dataset y genera IDs anónimos.
4. `make b1` genera runs dense globales.
5. `make b3` genera runs late-interaction.
6. `make b4` genera reranking VLM con traces JSONL.
7. `make b5` genera reranking agentic con traces JSONL.
8. `make eval` calcula métricas.
9. `make report` genera informe final.
10. `make serve` lanza la herramienta web.
11. `make slides` genera una presentación Markdown.
12. Los tests de fuga de información pasan.
13. Los runfiles tienen formato TREC válido.
14. Los traces cumplen JSON Schema.
15. Todas las métricas quedan registradas en MLflow.
16. El README permite reproducir el experimento desde cero.

---

# 20. README mínimo

El README debe incluir:

```text
# Instrument Retrieval Lab

## Goal
## Dataset setup
## Hardware requirements
## Docker setup
## Running a smoke test
## Running full experiments
## Viewing MLflow
## Launching the search tool
## Generating reports
## Generating slides
## Reproducibility notes
## Leakage prevention
## Citation
```

---

# 21. Smoke test

Implementar un smoke test con pocas imágenes:

```bash
make smoke
```

Debe:

```text
- procesar 50 imágenes
- generar qrels para 2 instrumentos
- ejecutar un modelo ligero
- producir un runfile
- calcular Recall@10
- validar trace schema si se usa reranker
```

El smoke test debe ejecutarse en menos de 10 minutos en una GPU razonable.

---

# 22. Limitaciones que debe documentar el informe

Incluir explícitamente:

```text
- Los modelos fundacionales pueden tener conocimiento previo de instrumentos comunes.
- No se puede garantizar que modelos cerrados no hayan visto imágenes públicas.
- Para reducir riesgo, se priorizan modelos open-weight y ejecución offline.
- No se usan nombres de archivo ni metadata contaminada.
- El agente no accede a internet.
- B4/B5 dependen del recall inicial: si el primer retriever no recupera un positivo en top-N, el reranker no puede recuperarlo.
- El coste computacional de B5 puede no compensar si B4 ya es fuerte.
```

---

# 23. Roadmap sugerido

## Fase 1 — Infraestructura

```text
Docker
CLI
dataset parser
anonimización
qrels
smoke test
```

## Fase 2 — B1

```text
OpenCLIP/JinaCLIP/BGE-VL/Qwen embeddings
FAISS
runs
métricas
```

## Fase 3 — B3

```text
ColPali/ColQwen
multivector retrieval
comparación con B1
```

## Fase 4 — B4

```text
VLM pointwise reranker
JSON schema
traces
evaluación
```

## Fase 5 — B5

```text
agentic deterministic workflow
caption
crops
crop VQA
evidence fusion
ablaciones
```

## Fase 6 — Tool + report

```text
Gradio/Streamlit
MLflow dashboard
report Markdown/HTML
slides
error analysis
```

---

# 24. Decisiones técnicas por defecto

Usar por defecto:

```text
Python 3.11
PyTorch CUDA
Transformers
OpenCLIP
FAISS
ColPali/ColQwen
Qwen2.5-VL como VLM inicial
LangGraph para B5
ranx para evaluación
MLflow para tracking
DVC para pipelines
Gradio para demo
Slidev para presentación final
Docker Compose para ejecución
```

---

# 25. Comando final esperado

Al terminar, esto debería funcionar:

```bash
git clone <repo>
cd instrument-ir-agentic

cp .env.example .env

make build
make check-gpu

# colocar dataset en data/raw/portuguese_instruments

make prepare
make smoke

make b1
make b3
make b4
make b5

make eval
make report
make slides
make serve
```

---

# 26. Entregables finales

```text
1. Código fuente completo.
2. Dockerfile y docker-compose GPU.
3. Scripts Make reproducibles.
4. Dataset procesado con IDs anónimos.
5. Qrels por split.
6. Runfiles TREC.
7. Métricas JSON/CSV/Markdown/LaTeX.
8. Trazas JSONL B4/B5.
9. Figuras de resultados.
10. Informe final.
11. Presentación en Markdown.
12. Demo web.
13. README reproducible.
14. Tests automáticos.
```

---

# 27. Primera tarea para el agente de código

Empieza implementando únicamente:

```text
- estructura del repositorio
- Docker GPU base
- CLI Typer
- parser COCO
- anonimización
- generación de qrels
- runfile TREC dummy
- evaluación dummy con ranx
- smoke test
```

No implementes todavía modelos pesados hasta que el smoke test pase.

Después implementa B1, luego B3, luego B4 y finalmente B5.

[1]: https://data.mendeley.com/datasets/pk7txkgt4v/2 "Visual Dataset of Traditional Portuguese Musical Instruments - Mendeley Data"
[2]: https://github.com/illuin-tech/colpali "GitHub - illuin-tech/colpali: The code used to train and run inference with the ColVision models, e.g. ColPali, ColQwen2, and ColSmol. · GitHub"
[3]: https://github.com/DataArcTech/RagVL "GitHub - DataArcTech/RagVL: Official PyTorch Implementation of MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training. · GitHub"
[4]: https://docs.langchain.com/oss/python/langgraph/workflows-agents "Workflows and agents - Docs by LangChain"
[5]: https://doc.dvc.org/user-guide "User Guide | Data Version Control · DVC"
[6]: https://revealjs.com/ "The HTML presentation framework | reveal.js"
