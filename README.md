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
- **Comparativa Qwen3.6/MIDA**: reranking zero-shot top-100 con `Qwen3.6-27B-GGUF`
  servido por llama.cpp multimodal en MIDA. Resultados en
  `results/mida-qwen36-27b/2026-06-30_zero_shot_top100/`.
- **Presentación Slidev**: `slides/slidev/`, titulada
  *Multimodal Visual Information Retrieval of Traditional Portuguese Instruments: A Reproducible
  Comparison of Dense Retrieval and Agentic Reranking*.

## Dataset y corpus

Dataset: *Visual Dataset of Traditional Portuguese Musical Instruments* (Mendeley Data,
DOI `10.17632/pk7txkgt4v.2`, CC BY 4.0). El artículo asociado es
*Comprehensive dataset of Portuguese folk instruments for computer vision and heritage research*,
publicado en *Data in Brief* 61, 2025, Art. no. 111739, DOI `10.1016/j.dib.2025.111739`.

El corpus procede de frames extraídos de vídeos de
[*A Música Portuguesa a Gostar Dela Própria*](https://amusicaportuguesaagostardelapropria.org/map),
un archivo audiovisual de música tradicional portuguesa. En este repositorio se formula como un
problema de recuperación visual: la consulta es el nombre de un instrumento y cada documento es una
imagen/frame del dataset.

Coloca el dataset en:

```
data/raw/portuguese_instruments/{train,valid,test}/_annotations.coco.json + imágenes
```

22 instrumentos (la categoría COCO `instruments` id 0 es paraguas y se excluye). Los nombres de
archivo contienen el instrumento y el id de Vimeo → se anonimizan; el mapping privado
(`data/processed/image_id_mapping.parquet`) solo se usa para cargar píxeles y para el buscador.

## Modelos evaluados y publicaciones asociadas

| Sistema | Implementación en el repo | Modelos usados | Publicación / fuente asociada |
|---|---|---|---|
| B1 dense global | `instrument-ir retrieve --model openclip-vitb32` y `openclip-vitl14` | OpenCLIP `ViT-B-32` (`laion2b_s34b_b79k`) y `ViT-L-14` (`laion2b_s32b_b82k`) | CLIP: Radford et al. 2021; OpenCLIP: Ilharco et al. / MLFoundations. |
| B1 dense multilingüe | `instrument-ir retrieve --model jinaclip` | `jinaai/jina-clip-v2` | Jina CLIP: Jina AI, 2024, arXiv `2405.20204`; model card de Jina AI. |
| B3 late interaction | `instrument-ir retrieve --model colqwen` | `vidore/colqwen2-v1.0` | Familia ColPali/ColQwen: Faysse et al. 2024, arXiv `2407.01449`; base Qwen2-VL: Wang et al. 2024, arXiv `2409.12191`. |
| B4 VLM reranking | `instrument-ir rerank-vlm` sobre candidatos dense | Run GPU full: `Qwen/Qwen2.5-VL-3B-Instruct` servido como `qwen2.5-vl`; run MIDA: `unsloth/Qwen3.6-27B-GGUF` con `mmproj-F16.gguf` | Qwen2.5-VL Technical Report: Qwen Team, 2025, arXiv `2502.13923`; Qwen3.6 se documenta mediante model card GGUF de Unsloth. |
| B5 agentic reranking | `instrument-ir rerank-agent` con grafo propio determinista | Mismo VLM que B4 en el run correspondiente; añade inspección full-image, crops deterministas y fusión de evidencias | Inspirado metodológicamente por ReAct: Yao et al. 2023, arXiv `2210.03629`; el agente de este repo no usa LangGraph. |

Notas de interpretación:

- B1/B3 generan rankings directamente sobre todo el split de test.
- B4/B5 no buscan en todo el corpus: reordenan candidatos producidos por un dense retriever.
- Qwen3.6 top-100 usa `openclip-vitl14` como generador de candidatos y reordena los 100 primeros.
- vLLM y llama.cpp son infraestructura de serving, no sistemas de ranking por sí mismos.

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

## Referencias

[1] A Música Portuguesa a Gostar Dela Própria, “Mapa,” accessed Jun. 30, 2026. [Online].
Available: https://amusicaportuguesaagostardelapropria.org/map

[2] N. Zendron *et al.*, “Comprehensive dataset of Portuguese folk instruments for computer
vision and heritage research,” *Data in Brief*, vol. 61, Art. no. 111739, 2025,
doi: `10.1016/j.dib.2025.111739`.

[3] N. Zendron *et al.*, “Portuguese folk instruments dataset,” Mendeley Data, V2, 2025,
doi: `10.17632/pk7txkgt4v.2`.

[4] A. Radford *et al.*, “Learning transferable visual models from natural language supervision,”
in *Proc. ICML*, 2021.

[5] G. Ilharco, M. Wortsman, R. Wightman, C. Gordon, N. Carlini, R. Taori, A. Dave, V. Shankar,
H. Namkoong, J. Miller, H. Hajishirzi, A. Farhadi, and L. Schmidt, “OpenCLIP,” Zenodo, 2021,
doi: `10.5281/zenodo.5143773`.

[6] Jina AI, “Jina CLIP: Your CLIP model is also your text retriever,” arXiv:2405.20204, 2024.

[7] M. Faysse *et al.*, “ColPali: Efficient document retrieval with vision language models,”
arXiv:2407.01449, 2024.

[8] P. Wang *et al.*, “Qwen2-VL: Enhancing vision-language model's perception of the world at
any resolution,” arXiv:2409.12191, 2024.

[9] Qwen Team, “Qwen2.5-VL technical report,” arXiv:2502.13923, 2025.

[10] T. Yao *et al.*, “ReAct: Synergizing reasoning and acting in language models,” in
*Proc. ICLR*, 2023.

[11] G. Gerganov, “llama.cpp,” GitHub repository, 2023. [Online]. Available:
https://github.com/ggml-org/llama.cpp

[12] Unsloth, “Qwen3.6-27B-GGUF,” Hugging Face model card, accessed Jun. 30, 2026. [Online].
Available: https://huggingface.co/unsloth/Qwen3.6-27B-GGUF
