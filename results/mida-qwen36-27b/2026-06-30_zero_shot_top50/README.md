# Qwen3.6-27B zero-shot VLM reranking

Snapshot from the Qwen3.6-27B multimodal comparison run, executed from
`esalab-big` against the MIDA llama.cpp endpoint on the Tailnet.

## Run setup

- Date: 2026-06-30
- Remote runner: `esalab-big`
- VLM endpoint from runner: `http://100.127.120.42:8080/v1`
- Served model alias: `qwen36-27b`
- Model source: `unsloth/Qwen3.6-27B-GGUF`
- GGUF weights: `Qwen3.6-27B-Q4_K_M.gguf`
- Multimodal projector: `mmproj-F16.gguf`
- Thinking disabled: `VLM_DISABLE_THINKING=true`
- Dense candidate source: `openclip-vitl14`
- Candidate depth: `TOPN=50`
- Final depth: `FINAL_K=50`
- Split: `test`
- Queries: 66
- Candidates reranked: 3300
- Exit code: 0

## Key metrics

| metric | value |
|---|---:|
| recall@10 | 0.0358 |
| recall@20 | 0.0515 |
| recall@50 | 0.1052 |
| recall@100 | 0.1052 |
| precision@10 | 0.3394 |
| ndcg@10 | 0.3642 |
| ndcg@20 | 0.3102 |
| ndcg@100 | 0.1847 |
| map | 0.0559 |
| mrr | 0.5296 |

## Reranking diagnostics

| diagnostic | value |
|---|---:|
| candidate_recall@50 | 0.1052 |
| oracle_recall@50 | 0.1052 |
| rerank_gain@50 | 0.0000 |
| delta_ndcg@50 | 0.0284 |
| delta_map | 0.0075 |

## Interpretation

This is a constrained top-50 zero-shot comparison, so recall@100 is capped by
the input candidate set and is not directly comparable with the earlier top-200
B4/B5 runs. The run does improve ordering within the available candidates:
`delta_ndcg@50` is positive and `mrr` is high, but the dense top-50 candidate
ceiling limits recall.

## Artifacts

- `outputs/runs/B4_qwen36_zero_shot_test.trec`
- `outputs/runs/DENSE_qwen36_test.trec`
- `outputs/candidates/B4_qwen36_zero_shot_test.parquet`
- `outputs/rerank_traces/B4_qwen36_zero_shot_test.jsonl`
- `outputs/metrics/B4_qwen36_zero_shot_test.json`
- `outputs/metrics/B4_qwen36_zero_shot_test__rerankmetrics.json`
- `outputs/reports/final_report.md`
- `outputs/remote/qwen36_zero_shot.log`
- `outputs/remote/qwen36_zero_shot.exit`
