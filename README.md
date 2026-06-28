# Instrument Retrieval Lab

Framework experimental reproducible de **recuperación visual de imágenes** (visual Information
Retrieval) para instrumentos tradicionales portugueses. Dada una query textual ("encuéntrame
registros con adufe"), recupera y ordena las imágenes que contienen ese instrumento, **sin usar
nombres de archivo, rutas ni etiquetas durante la inferencia**. Compara los sistemas del ADR:
B1 (dense global), B3 (late-interaction), B4 (dense + VLM reranker), B5 (dense + agente reranker).

Ver el PRD completo en [ADR.md](ADR.md).

## Estado

- **Fase 1 (infraestructura) — COMPLETA**: parser COCO, anonimización, qrels, retrieve dummy,
  evaluación con ranx, smoke test y tests anti-fuga.
- Fases 2–6 (B1/B3/B4/B5, buscador, informes, slides) — pendientes.

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

## Citation

Dataset: *Visual Dataset of Traditional Portuguese Musical Instruments*, Mendeley Data,
DOI `10.17632/pk7txkgt4v.2`.
