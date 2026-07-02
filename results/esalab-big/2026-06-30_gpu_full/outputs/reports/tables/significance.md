# Significancia estadística (n=66 consultas, bootstrap 10k, permutación pareada 10k)

IC 95% del delta por remuestreo de consultas. p ajustado por Holm-Bonferroni dentro de cada métrica.


## recall@100

| comparación | delta | IC 95% | p (Holm) | signif. |
|---|---|---|---|---|
| JinaCLIP (dense) vs OpenCLIP L/14 (dense) | +0.0130 | [-0.022, +0.049] | 1.000 | no |
| ColQwen (late-interaction) vs OpenCLIP L/14 (dense) | -0.0192 | [-0.056, +0.017] | 1.000 | no |
| VLM-rerank (Qwen2.5-VL-3B) vs OpenCLIP L/14 (dense) | +0.0096 | [+0.000, +0.022] | 0.422 | no |
| Agentic vs VLM-rerank (Qwen2.5-VL-3B) | +0.0022 | [-0.002, +0.006] | 1.000 | no |
| VLM-rerank (Qwen3.6-27B, top-100) vs OpenCLIP L/14 (dense) | +0.0000 | [+0.000, +0.000] | 1.000 | no |
| VLM-rerank (Qwen3.6-27B, top-100) vs VLM-rerank (Qwen2.5-VL-3B) | -0.0096 | [-0.022, -0.000] | 0.422 | no |

## ndcg@10

| comparación | delta | IC 95% | p (Holm) | signif. |
|---|---|---|---|---|
| JinaCLIP (dense) vs OpenCLIP L/14 (dense) | +0.0016 | [-0.063, +0.063] | 1.000 | no |
| ColQwen (late-interaction) vs OpenCLIP L/14 (dense) | -0.0530 | [-0.124, +0.014] | 0.519 | no |
| VLM-rerank (Qwen2.5-VL-3B) vs OpenCLIP L/14 (dense) | +0.0099 | [-0.041, +0.060] | 1.000 | no |
| Agentic vs VLM-rerank (Qwen2.5-VL-3B) | +0.0020 | [-0.031, +0.034] | 1.000 | no |
| VLM-rerank (Qwen3.6-27B, top-100) vs OpenCLIP L/14 (dense) | +0.1384 | [+0.076, +0.204] | 0.001 | **sí** |
| VLM-rerank (Qwen3.6-27B, top-100) vs VLM-rerank (Qwen2.5-VL-3B) | +0.1285 | [+0.076, +0.186] | 0.001 | **sí** |

## ndcg@100

| comparación | delta | IC 95% | p (Holm) | signif. |
|---|---|---|---|---|
| JinaCLIP (dense) vs OpenCLIP L/14 (dense) | +0.0184 | [-0.016, +0.056] | 0.849 | no |
| ColQwen (late-interaction) vs OpenCLIP L/14 (dense) | -0.0187 | [-0.061, +0.021] | 0.849 | no |
| VLM-rerank (Qwen2.5-VL-3B) vs OpenCLIP L/14 (dense) | +0.0097 | [-0.007, +0.027] | 0.849 | no |
| Agentic vs VLM-rerank (Qwen2.5-VL-3B) | +0.0086 | [-0.000, +0.018] | 0.292 | no |
| VLM-rerank (Qwen3.6-27B, top-100) vs OpenCLIP L/14 (dense) | +0.0261 | [+0.011, +0.041] | 0.006 | **sí** |
| VLM-rerank (Qwen3.6-27B, top-100) vs VLM-rerank (Qwen2.5-VL-3B) | +0.0164 | [+0.001, +0.032] | 0.209 | no |

## map

| comparación | delta | IC 95% | p (Holm) | signif. |
|---|---|---|---|---|
| JinaCLIP (dense) vs OpenCLIP L/14 (dense) | +0.0083 | [-0.010, +0.029] | 1.000 | no |
| ColQwen (late-interaction) vs OpenCLIP L/14 (dense) | -0.0055 | [-0.027, +0.016] | 1.000 | no |
| VLM-rerank (Qwen2.5-VL-3B) vs OpenCLIP L/14 (dense) | +0.0005 | [-0.009, +0.010] | 1.000 | no |
| Agentic vs VLM-rerank (Qwen2.5-VL-3B) | +0.0031 | [-0.001, +0.007] | 0.518 | no |
| VLM-rerank (Qwen3.6-27B, top-100) vs OpenCLIP L/14 (dense) | +0.0123 | [+0.003, +0.021] | 0.046 | **sí** |
| VLM-rerank (Qwen3.6-27B, top-100) vs VLM-rerank (Qwen2.5-VL-3B) | +0.0118 | [+0.004, +0.019] | 0.024 | **sí** |

## mrr

| comparación | delta | IC 95% | p (Holm) | signif. |
|---|---|---|---|---|
| JinaCLIP (dense) vs OpenCLIP L/14 (dense) | -0.0156 | [-0.108, +0.074] | 1.000 | no |
| ColQwen (late-interaction) vs OpenCLIP L/14 (dense) | -0.0395 | [-0.143, +0.062] | 1.000 | no |
| VLM-rerank (Qwen2.5-VL-3B) vs OpenCLIP L/14 (dense) | +0.0339 | [-0.047, +0.113] | 1.000 | no |
| Agentic vs VLM-rerank (Qwen2.5-VL-3B) | +0.0451 | [-0.022, +0.116] | 0.848 | no |
| VLM-rerank (Qwen3.6-27B, top-100) vs OpenCLIP L/14 (dense) | +0.1515 | [+0.050, +0.256] | 0.024 | **sí** |
| VLM-rerank (Qwen3.6-27B, top-100) vs VLM-rerank (Qwen2.5-VL-3B) | +0.1175 | [+0.035, +0.205] | 0.039 | **sí** |
