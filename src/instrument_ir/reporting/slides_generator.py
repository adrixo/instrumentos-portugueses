"""Generador de slides en Marp (ADR §14). Markdown puro, sin dependencias."""

from __future__ import annotations

from pathlib import Path

from .tables import load_all_metrics, macro_table_md

MARP_HEADER = """---
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
"""


def generate_slides(
    metrics_dir: Path = Path("outputs/metrics"),
    out_path: Path = Path("outputs/slides/slides.md"),
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = load_all_metrics(metrics_dir)
    table = macro_table_md(all_metrics) if all_metrics else "_(sin métricas)_"

    deck = (
        MARP_HEADER
        + "\n## Resultados principales (macro)\n\n"
        + table
        + "\n\n---\n\n## Conclusiones\n"
        + "- El reranker (B4/B5) mejora donde el dense falla (instrumentos pequeños/parecidos).\n"
        + "- Coste vs calidad: B5 solo compensa si B4 no basta.\n\n---\n\n## Trabajo futuro\n"
        + "- Crops por saliencia (GroundingDINO/SAM), más modelos dense, test final.\n"
    )
    out_path.write_text(deck, encoding="utf-8")
    return out_path
