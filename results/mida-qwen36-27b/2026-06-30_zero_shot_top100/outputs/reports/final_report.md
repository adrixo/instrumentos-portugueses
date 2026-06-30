# Instrument Retrieval Lab — Informe de resultados

## 1. Objetivo
Recuperación visual de instrumentos tradicionales portugueses: dada una query textual, ordenar las
imágenes que contienen el instrumento, sin usar nombres de archivo, rutas ni etiquetas en inferencia.

## 2. Dataset
22 instrumentos (categoría paraguas excluida), splits train/valid/test, ground truth COCO
multi-etiqueta. DOI Mendeley 10.17632/pk7txkgt4v.2.

## 3. Prevención de fuga
image_id anónimos, mapping privado, prompts/traces sin filename/vimeo/labels. Tests automáticos.

## 4. Métodos
B1 dense global · B3 late-interaction (ColQwen) · B4 dense+VLM reranker · B5 dense+agente determinista.

## 5. Resultados (macro por instrumento)

| system | recall@20 | recall@50 | recall@100 | ndcg@10 | ndcg@100 | map | mrr |
|---|---|---|---|---|---|---|---|
| B1_jinaclip_test | 0.0466 | 0.1145 | 0.1938 | 0.2517 | 0.2640 | 0.0842 | 0.3282 |
| B1_jinaclip_valid | 0.0469 | 0.0947 | 0.1693 | 0.2323 | 0.2365 | 0.0723 | 0.2740 |
| B1_openclip-vitb32_test | 0.0242 | 0.0776 | 0.1485 | 0.1562 | 0.2007 | 0.0514 | 0.2251 |
| B1_openclip-vitb32_valid | 0.0542 | 0.1036 | 0.1787 | 0.1865 | 0.2219 | 0.0650 | 0.2841 |
| B1_openclip-vitl14_test | 0.0530 | 0.1052 | 0.1808 | 0.2501 | 0.2456 | 0.0760 | 0.3438 |
| B1_openclip-vitl14_valid | 0.0454 | 0.1130 | 0.2169 | 0.2646 | 0.2555 | 0.0754 | 0.3611 |
| B3_colqwen_test | 0.0326 | 0.0872 | 0.1617 | 0.1972 | 0.2268 | 0.0705 | 0.3043 |
| B3_colqwen_valid | 0.0320 | 0.0802 | 0.1809 | 0.1867 | 0.2168 | 0.0680 | 0.2466 |
| B4_qwen36_zero_shot_test | 0.0515 | 0.1052 | 0.1052 | 0.3642 | 0.1847 | 0.0559 | 0.5296 |
| B4_qwen36_zero_shot_test__rerankmetrics | — | — | — | — | — | — | — |
| B4_qwen36_zero_shot_top100_test | 0.0629 | 0.1111 | 0.1808 | 0.3885 | 0.2717 | 0.0882 | 0.4953 |
| B4_qwen36_zero_shot_top100_test__rerankmetrics | — | — | — | — | — | — | — |
| B4_test | 0.0419 | 0.1027 | 0.1904 | 0.2600 | 0.2553 | 0.0764 | 0.3777 |
| B4_test__rerankmetrics | — | — | — | — | — | — | — |
| B5_full_image_only_test | 0.0419 | 0.1027 | 0.1904 | 0.2600 | 0.2553 | 0.0764 | 0.3777 |
| B5_full_test | 0.0462 | 0.1094 | 0.1926 | 0.2620 | 0.2639 | 0.0795 | 0.4229 |
| B5_full_test__rerankmetrics | — | — | — | — | — | — | — |
| B5_max_score_only_test | 0.0462 | 0.1094 | 0.1926 | 0.2620 | 0.2639 | 0.0795 | 0.4229 |
| B5_no_caption_test | 0.0462 | 0.1094 | 0.1926 | 0.2620 | 0.2639 | 0.0795 | 0.4229 |
| B5_no_crops_test | 0.0419 | 0.1027 | 0.1904 | 0.2600 | 0.2553 | 0.0764 | 0.3777 |
| B5_weighted_fusion_test | 0.0479 | 0.1110 | 0.1853 | 0.2727 | 0.2597 | 0.0787 | 0.3885 |
| query_latency_summary | — | — | — | — | — | — | — |

![Recall@K](figures/recall_at_k.png)

## 6. Resultados por instrumento (Recall@100)

| instrument | B1_jinaclip_test | B1_jinaclip_valid | B1_openclip-vitb32_test | B1_openclip-vitb32_valid | B1_openclip-vitl14_test | B1_openclip-vitl14_valid | B3_colqwen_test | B3_colqwen_valid | B4_qwen36_zero_shot_test | B4_qwen36_zero_shot_top100_test | B4_test | B5_full_image_only_test | B5_full_test | B5_max_score_only_test | B5_no_caption_test | B5_no_crops_test | B5_weighted_fusion_test | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adufe | 0.088 | 0.029 | 0.092 | 0.059 | 0.077 | 0.082 | 0.053 | 0.044 | 0.024 | 0.077 | 0.046 | 0.046 | 0.064 | 0.064 | 0.064 | 0.046 | 0.059 | B1_openclip-vitb32_test |
| bombos | 0.266 | 0.304 | 0.195 | 0.182 | 0.205 | 0.206 | 0.243 | 0.237 | 0.108 | 0.205 | 0.245 | 0.245 | 0.252 | 0.252 | 0.252 | 0.245 | 0.255 | B1_jinaclip_valid |
| caixa-tamboril | 0.378 | 0.341 | 0.211 | 0.216 | 0.256 | 0.243 | 0.155 | 0.159 | 0.138 | 0.256 | 0.288 | 0.288 | 0.305 | 0.305 | 0.305 | 0.288 | 0.319 | B1_jinaclip_test |
| castanholas | 0.217 | 0.152 | 0.150 | 0.061 | 0.183 | 0.000 | 0.050 | 0.030 | 0.150 | 0.183 | 0.200 | 0.200 | 0.217 | 0.217 | 0.217 | 0.200 | 0.200 | B1_jinaclip_test |
| cavaquinho | 0.105 | 0.082 | 0.087 | 0.059 | 0.119 | 0.098 | 0.065 | 0.041 | 0.073 | 0.119 | 0.107 | 0.107 | 0.121 | 0.121 | 0.121 | 0.107 | 0.097 | B5_full_test |
| concertina | 0.432 | 0.424 | 0.438 | 0.426 | 0.491 | 0.487 | 0.456 | 0.456 | 0.251 | 0.491 | 0.492 | 0.492 | 0.492 | 0.492 | 0.492 | 0.492 | 0.474 | B4_test |
| ferrinhos-triangulo | 0.240 | 0.038 | 0.213 | 0.256 | 0.147 | 0.282 | 0.187 | 0.115 | 0.053 | 0.147 | 0.160 | 0.160 | 0.160 | 0.160 | 0.160 | 0.160 | 0.147 | B1_openclip-vitl14_valid |
| flauta | 0.197 | 0.160 | 0.200 | 0.105 | 0.210 | 0.206 | 0.507 | 0.441 | 0.110 | 0.210 | 0.223 | 0.223 | 0.227 | 0.227 | 0.227 | 0.223 | 0.223 | B3_colqwen_test |
| gaita-de-foles | 0.220 | 0.245 | 0.180 | 0.194 | 0.170 | 0.196 | 0.206 | 0.218 | 0.113 | 0.170 | 0.172 | 0.172 | 0.184 | 0.184 | 0.184 | 0.172 | 0.192 | B1_jinaclip_valid |
| guitarra-portuguesa | 0.161 | 0.208 | 0.102 | 0.125 | 0.136 | 0.156 | 0.161 | 0.134 | 0.055 | 0.136 | 0.133 | 0.133 | 0.121 | 0.121 | 0.121 | 0.133 | 0.133 | B1_jinaclip_valid |
| matracas | 0.000 | 0.167 | 0.000 | 0.200 | 0.000 | 0.367 | 0.000 | 0.467 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | B3_colqwen_valid |
| palheta | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | B1_openclip-vitl14_valid |
| rabeca-chuleira | 0.235 | 0.228 | 0.220 | 0.123 | 0.220 | 0.368 | 0.152 | 0.140 | 0.205 | 0.220 | 0.242 | 0.242 | 0.250 | 0.250 | 0.250 | 0.242 | 0.212 | B1_openclip-vitl14_valid |
| reque-reque | 0.089 | 0.104 | 0.000 | 0.042 | 0.000 | 0.125 | 0.022 | 0.083 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | B1_openclip-vitl14_valid |
| sarronca | 0.000 | 0.367 | 0.000 | 0.500 | 0.000 | 0.367 | 0.056 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | B1_openclip-vitb32_valid |
| viola-amarantina | 0.412 | 0.114 | 0.137 | 0.144 | 0.281 | 0.227 | 0.288 | 0.295 | 0.203 | 0.281 | 0.314 | 0.314 | 0.314 | 0.314 | 0.314 | 0.314 | 0.314 | B1_jinaclip_test |
| viola-beiroa | 0.690 | 0.462 | 0.381 | 0.590 | 0.667 | 0.256 | 0.310 | 0.231 | 0.476 | 0.667 | 0.762 | 0.762 | 0.738 | 0.738 | 0.738 | 0.762 | 0.738 | B4_test |
| viola-braguesa | 0.098 | 0.137 | 0.126 | 0.083 | 0.126 | 0.226 | 0.126 | 0.119 | 0.046 | 0.126 | 0.115 | 0.115 | 0.115 | 0.115 | 0.115 | 0.115 | 0.115 | B1_openclip-vitl14_valid |
| viola-campanica | 0.070 | 0.068 | 0.119 | 0.106 | 0.129 | 0.129 | 0.169 | 0.136 | 0.051 | 0.129 | 0.112 | 0.112 | 0.099 | 0.099 | 0.099 | 0.112 | 0.094 | B3_colqwen_test |
| viola-de-arame | 0.152 | 0.011 | 0.295 | 0.376 | 0.276 | 0.258 | 0.200 | 0.032 | 0.162 | 0.276 | 0.276 | 0.276 | 0.276 | 0.276 | 0.276 | 0.276 | 0.276 | B1_openclip-vitb32_valid |
| viola-toeira | 0.000 | 0.000 | 0.037 | 0.000 | 0.222 | 0.091 | 0.074 | 0.000 | 0.074 | 0.222 | 0.222 | 0.222 | 0.222 | 0.222 | 0.222 | 0.222 | 0.148 | B1_openclip-vitl14_test |
| violao | 0.047 | 0.086 | 0.083 | 0.086 | 0.064 | 0.067 | 0.077 | 0.098 | 0.023 | 0.064 | 0.079 | 0.079 | 0.080 | 0.080 | 0.080 | 0.079 | 0.079 | B3_colqwen_valid |

## 7. Significancia estadística (gain vs baseline, Recall@100)

_(sin pares comparables)_

## 8. Sistemas evaluados

22 runs en `outputs/metrics`.

## 9. Limitaciones
- Modelos fundacionales pueden conocer instrumentos comunes.
- ~22 clases → potencia estadística limitada; CIs anchos.
- B4/B5 dependen del recall inicial del dense (oracle_recall@N).
