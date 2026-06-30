# Instrument Retrieval Lab

Framework experimental reproducible de **recuperación visual de imágenes** (visual Information
Retrieval) para instrumentos tradicionales portugueses. Dada una query textual ("encuéntrame
registros con adufe"), recupera y ordena las imágenes que contienen ese instrumento, **sin usar
nombres de archivo, rutas ni etiquetas durante la inferencia**. Compara los sistemas del ADR:
B1 (dense global), B3 (late-interaction), B4 (dense + VLM reranker), B5 (dense + agente reranker).

Ver el PRD completo en [ADR.md](ADR.md).

## Estado

Los 5 sistemas del ADR (B1/B3/B4/B5) + buscador, informes y slides están **implementados**.
- **Validado en CPU**: pipeline de datos, anti-fuga, B1 dense (real en valid), núcleos de
  B3 (MaxSim), B4 (reranker VLM) y B5 (agente) con backend mock + tests; informe y slides.
- **Validado en GPU**: run completo en `esalab-big` con B1/B3 en valid+test y B4/B5 sobre test.
  Resultados versionados en `results/esalab-big/2026-06-30_gpu_full/`.
- **Presentación Slidev**: `slides/slidev/`, titulada
  *Multimodal Visual Information Retrieval of Traditional Portuguese Instruments: A Reproducible
  Comparison of Dense Retrieval and Agentic Reranking*.

## Dataset setup

Dataset: *Visual Dataset of Traditional Portuguese Musical Instruments* (Mendeley Data,
DOI `10.17632/pk7txkgt4v.2`, CC BY 4.0). Colócalo en:

```
data/raw/portuguese_instruments/{train,valid,test}/_annotations.coco.json + imágenes
```

22 instrumentos (la categoría COCO `instruments` id 0 es paraguas y se excluye). Los nombres de
archivo contienen el instrumento y el id de Vimeo → se anonimizan; el mapping privado
(`data/processed/image_id_mapping.parquet`) solo se usa para cargar píxeles y para el buscador.

## Requisitos

- Python ≥ 3.10. Para B1+ se requiere GPU (extra `[dense]`); B4/B5 requieren GPU + servidor VLM.
- La Fase 1 corre en CPU.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # base (datos + evaluación)
# pip install -e ".[dense]"   # + OpenCLIP/torch/faiss (Fase 2)
```

## Smoke test (<10 min)

```bash
make smoke      # ~50 imágenes, 2 instrumentos, runfile, Recall@10
make test       # tests, incl. anti-fuga
```

## Flujo Fase 1

```bash
make prepare                    # parsea COCO, anonimiza, genera corpus/mapping/queries
make qrels SPLIT=valid          # ground truth TREC
make retrieve SPLIT=valid       # runfile dummy (placeholder de B1)
make eval SPLIT=valid           # métricas IR (Recall/nDCG/mAP/MRR + macro por instrumento)
# o todo junto:
make repro SPLIT=valid
```

## Docker GPU

```bash
make build
make check-gpu
docker compose -f docker/docker-compose.gpu.yml up
```

## Reproducibilidad

Seeds fijos (`utils/reproducibility.py`), `temperature=0` en VLM, trazas estructuradas (B4/B5),
métricas en MLflow (file-store), pipeline en `dvc.yaml`. Dataset inmutable por DOI.

## Prevención de fuga

`tests/test_no_filename_leakage.py` verifica que los artefactos públicos (corpus, runfile, qrels) no
contienen nombres de archivo, ids de Vimeo ni etiquetas, y que los `image_id` son anónimos.

## Troubleshooting / notas

- **faiss + torch en macOS (deadlock OpenMP)**: si al lanzar retrieval con OpenCLIP el proceso se
  queda colgado (CPU ~0%, sin avanzar) o aparece `OMP: Error #15`, es el choque de dos runtimes
  OpenMP (faiss y torch). En Mac, exporta `INSTRUMENT_IR_NO_FAISS=1` para usar el camino numpy exacto
  (idéntico resultado; faiss solo aporta velocidad a gran escala, innecesaria con ~1.3k vectores). En
  la máquina GPU remota (Linux + faiss-cpu/gpu) no ocurre.
- **HF offline**: con `HF_HUB_OFFLINE=1` se evita contactar el Hub si los pesos ya están cacheados.

## Citation

Dataset: *Visual Dataset of Traditional Portuguese Musical Instruments*, Mendeley Data,
DOI `10.17632/pk7txkgt4v.2`.
