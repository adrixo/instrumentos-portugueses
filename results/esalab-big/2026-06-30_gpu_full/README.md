# Esalab-big GPU Full Run

Snapshot descargado desde `esalab-big` (`100.69.221.87`) el 2026-06-30.

- Proyecto remoto: `/home/esalab/Escritorio/instrumentos_portugueses_ir`
- GPU: NVIDIA GeForce RTX 3090 Ti 24GB
- Dataset: Mendeley `10.17632/pk7txkgt4v.2`
- Split evaluado para B4/B5: `test`
- `TOPN=200`, `FINAL_K=100`, `DENSE_MODEL=openclip-vitl14`
- VLM servido: `Qwen/Qwen2.5-VL-3B-Instruct` como `qwen2.5-vl`
- vLLM: `vllm/vllm-openai:v0.10.1.1`, `max_model_len=4096`, `gpu_memory_utilization=0.60`
- Reranking VLM: `VLM_MAX_IMAGE_SIDE=768`, `VLM_JPEG_QUALITY=85`, `VLM_WORKERS=8`, cache persistente activada

## Artefactos

- Reporte final: `outputs/reports/final_report.md`
- Reporte HTML: `outputs/reports/final_report.html`
- Tablas: `outputs/reports/tables/`
- Runfiles TREC: `outputs/runs/`
- Metricas JSON: `outputs/metrics/`
- Trazas B4/B5: `outputs/rerank_traces/`
- Logs: `outputs/remote/gpu_full.log`

## Resumen Test

| system | recall@100 | ndcg@100 | map | mrr |
|---|---:|---:|---:|---:|
| B1_openclip-vitb32_test | 0.1485 | 0.2007 | 0.0514 | 0.2251 |
| B1_openclip-vitl14_test | 0.1808 | 0.2456 | 0.0760 | 0.3438 |
| B1_jinaclip_test | 0.1938 | 0.2640 | 0.0842 | 0.3282 |
| B3_colqwen_test | 0.1617 | 0.2268 | 0.0705 | 0.3043 |
| B4_test | 0.1904 | 0.2553 | 0.0764 | 0.3777 |
| B5_full_test | 0.1926 | 0.2639 | 0.0795 | 0.4229 |
| B5_no_crops_test | 0.1904 | 0.2553 | 0.0764 | 0.3777 |
| B5_no_caption_test | 0.1926 | 0.2639 | 0.0795 | 0.4229 |
| B5_full_image_only_test | 0.1904 | 0.2553 | 0.0764 | 0.3777 |
| B5_max_score_only_test | 0.1926 | 0.2639 | 0.0795 | 0.4229 |
| B5_weighted_fusion_test | 0.1853 | 0.2597 | 0.0787 | 0.3885 |

## Reranking Gain

| system | candidate_recall@200 | oracle_recall@100 | rerank_gain@100 | delta_ndcg@100 | delta_map |
|---|---:|---:|---:|---:|---:|
| B4_test | 0.2938 | 0.2781 | 0.0096 | 0.0097 | -0.0309 |
| B5_full_test | 0.2938 | 0.2781 | 0.0118 | 0.0183 | -0.0278 |

## Query Latency

Per-query latency was benchmarked with corpus embeddings cached for B1/B3. B4/B5 latency is estimated
from trace timestamp spans over top-200 reranking runs.

| system | mean seconds/query | median seconds/query | p95 seconds/query |
|---|---:|---:|---:|
| openclip-vitl14 | 0.019 | 0.017 | 0.017 |
| jinaclip | 0.104 | 0.092 | 0.096 |
| colqwen | 0.479 | 0.476 | 0.495 |
| B4_VLM | 30.117 | 27.748 | 52.407 |
| B5_agentic | 44.885 | 29.677 | 118.313 |
