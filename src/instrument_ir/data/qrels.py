"""Construcción de qrels (ground truth IR) en formato TREC (ADR §5.3).

Formato:  query_id 0 image_id relevance
- relevance = 1 si la imagen contiene el instrumento de la query (GT COCO).
- Solo se escriben los relevantes (relevance > 0); los no relevantes no se listan.

El GT sale del mapping privado (`gt_instruments`), NUNCA del nombre de archivo. Una imagen
multi-etiqueta es relevante para varias queries (distintos instrumentos).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .queries import Query


def build_qrels(mapping: pd.DataFrame, queries: list[Query]) -> list[tuple[str, str, int]]:
    """Devuelve filas (query_id, image_id, relevance=1) para todos los pares relevantes."""
    # Índice instrumento -> conjunto de image_id que lo contienen.
    instrument_to_images: dict[str, list[str]] = {}
    for image_id, gt in zip(mapping["image_id"], mapping["gt_instruments"]):
        for instrument in gt:
            instrument_to_images.setdefault(instrument, []).append(image_id)

    rows: list[tuple[str, str, int]] = []
    for q in queries:
        for image_id in instrument_to_images.get(q.instrument_id, ()):
            rows.append((q.query_id, image_id, 1))
    return rows


def write_qrels_trec(rows: list[tuple[str, str, int]], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{qid} 0 {iid} {rel}" for qid, iid, rel in rows]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


def load_qrels_trec(path: Path) -> dict[str, dict[str, int]]:
    """Carga qrels TREC a dict {query_id: {image_id: relevance}} (formato ranx)."""
    qrels: dict[str, dict[str, int]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        qid, _zero, iid, rel = line.split()
        qrels.setdefault(qid, {})[iid] = int(rel)
    return qrels
