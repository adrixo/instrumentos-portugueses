# Qwen3.6-27B zero-shot VLM reranking, top-100

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
- Candidate depth: `TOPN=100`
- Final depth: `FINAL_K=100`
- Split: `test`
- Queries: 66
- Candidates reranked: 6600
- Reused top-50 judgements from `B4_qwen36_zero_shot_test`: 3300
- New top-51-to-100 judgements: 3300
- Exit code: 0

## Key metrics

| metric | value |
|---|---:|
| recall@10 | 0.0386 |
| recall@20 | 0.0629 |
| recall@50 | 0.1111 |
| recall@100 | 0.1808 |
| precision@10 | 0.3818 |
| ndcg@10 | 0.3885 |
| ndcg@20 | 0.3563 |
| ndcg@100 | 0.2717 |
| map | 0.0882 |
| mrr | 0.4953 |

## Reranking diagnostics

| diagnostic | value |
|---|---:|
| candidate_recall@100 | 0.1808 |
| oracle_recall@100 | 0.1808 |
| rerank_gain@100 | 0.0000 |
| delta_ndcg@100 | 0.0261 |
| delta_map | 0.0123 |

## Latency notes

The top-100 trace was resumed from the previous top-50 Qwen run, so the
observed trace contains reused judgements for candidates 1-50 and fresh
judgements for candidates 51-100.

| latency estimate | value |
|---|---:|
| Prior top-50 mean trace span | 154.9 s/query |
| Incremental candidates 51-100 mean trace span | 129.3 s/query |
| Fresh top-100 sequential estimate | 284.2 s/query |

## Interpretation

Moving from top-50 to top-100 removes the main recall caveat from the previous
Qwen comparison. Qwen3.6 reaches the same top-100 candidate ceiling as the
OpenCLIP L/14 dense run and improves ranking quality within that ceiling:
`delta_ndcg@100` and `delta_map` are both positive. JinaCLIP and the B5 agentic
run still recover slightly more relevant material at Recall@100, while Qwen3.6
is strongest in early ranking quality metrics.

## Artifacts

- `outputs/runs/B4_qwen36_zero_shot_top100_test.trec`
- `outputs/runs/DENSE_qwen36_top100_test.trec`
- `outputs/candidates/B4_qwen36_zero_shot_top100_test.parquet`
- `outputs/rerank_traces/B4_qwen36_zero_shot_top100_test.jsonl`
- `outputs/metrics/B4_qwen36_zero_shot_top100_test.json`
- `outputs/metrics/B4_qwen36_zero_shot_top100_test__rerankmetrics.json`
- `outputs/reports/final_report.md`
- `outputs/remote/qwen36_zero_shot.log`
- `outputs/remote/qwen36_zero_shot.exit`
- `outputs/remote/run_qwen_top100_resume.sh`
