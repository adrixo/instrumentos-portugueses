"""Lectura/escritura de runfiles TREC (ADR §4 B1).

Formato:  query_id Q0 image_id rank score run_name
"""

from __future__ import annotations

from pathlib import Path

from ..retrieval.base import ScoredDoc


def write_run_trec(
    rankings: dict[str, list[ScoredDoc]], run_name: str, out_path: Path
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for query_id, docs in rankings.items():
        for rank, doc in enumerate(docs, start=1):
            lines.append(
                f"{query_id} Q0 {doc.image_id} {rank} {doc.score:.6f} {run_name}"
            )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


def load_run_trec(path: Path) -> dict[str, dict[str, float]]:
    """Carga runfile a {query_id: {image_id: score}} (formato ranx)."""
    run: dict[str, dict[str, float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        query_id, _q0, image_id, _rank, score = parts[:5]
        run.setdefault(query_id, {})[image_id] = float(score)
    return run
