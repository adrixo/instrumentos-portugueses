---
marp: true
theme: default
paginate: true
---

# Instrument Retrieval Lab
### Recuperación visual de instrumentos tradicionales portugueses
B1 dense · B3 late-interaction · B4 VLM reranker · B5 agente determinista

---

## Motivación
- "Encuéntrame imágenes con adufe" → IR visual, sin trampas (sin filename/labels).
- Ground truth COCO multi-etiqueta, 22 instrumentos.

---

## Formulación como IR visual
- Query textual por instrumento (pt/en/es).
- Relevante = la imagen contiene la clase. Métricas: Recall/nDCG/mAP/MRR.

---

## Control de fuga
- image_id anónimos, mapping privado, prompts/traces sin filename/vimeo/labels.
- Tests automáticos de fuga.

---

## Resultados principales (macro)

| system | recall@20 | recall@50 | recall@100 | ndcg@10 | ndcg@100 | map | mrr |
|---|---|---|---|---|---|---|---|
| B1_dummy_valid | 0.0190 | 0.0439 | 0.0787 | 0.0762 | 0.0942 | 0.0093 | 0.2004 |
| B1_mini | 0.3673 | 0.5287 | 0.5287 | 0.5568 | 0.5249 | 0.3614 | 0.7153 |
| B1_openclip-vitb32_valid | 0.0559 | 0.1045 | 0.1787 | 0.1867 | 0.2233 | 0.0659 | 0.2927 |
| B1_openclip-vitb32_valid__rerankmetrics | — | — | — | — | — | — | — |
| B4_mini | 0.3907 | 0.5287 | 0.5287 | 0.6032 | 0.5669 | 0.3955 | 0.8667 |
| B4_mini__rerankmetrics | — | — | — | — | — | — | — |
| B5_full_mini | 0.4199 | 0.5287 | 0.5287 | 0.6225 | 0.5836 | 0.4175 | 0.9167 |
| B5_full_mini__rerankmetrics | — | — | — | — | — | — | — |
| B5_fullimg_mini | 0.3907 | 0.5287 | 0.5287 | 0.6032 | 0.5669 | 0.3955 | 0.8667 |
| B5_nocrops_mini | 0.3907 | 0.5287 | 0.5287 | 0.6032 | 0.5669 | 0.3955 | 0.8667 |

---

## Conclusiones
- El reranker (B4/B5) mejora donde el dense falla (instrumentos pequeños/parecidos).
- Coste vs calidad: B5 solo compensa si B4 no basta.

---

## Trabajo futuro
- Crops por saliencia (GroundingDINO/SAM), más modelos dense, test final.
