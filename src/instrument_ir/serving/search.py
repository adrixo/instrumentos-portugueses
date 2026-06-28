"""Lógica del buscador, desacoplada de Gradio (testeable sin UI). ADR §12.

Navega runfiles ya generados (B1/B3/B4/B5) y muestra los resultados con su id de Vimeo y enlace al
vídeo original. El mapping privado se usa SOLO aquí para mostrar (vimeo_id + ruta de imagen), nunca en
inferencia.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.qrels import load_qrels_trec
from ..utils.trec import load_run_trec


class SearchService:
    def __init__(
        self,
        runs_dir: Path = Path("outputs/runs"),
        mapping_path: Path = Path("data/processed/image_id_mapping.parquet"),
        raw_root: Path = Path("data/raw/portuguese_instruments"),
        qrels_dir: Path = Path("data/processed/qrels"),
    ):
        self.runs_dir = Path(runs_dir)
        self.raw_root = Path(raw_root)
        self.qrels_dir = Path(qrels_dir)
        mp = pd.read_parquet(mapping_path)
        self._map = {
            r.image_id: {"vimeo_id": r.vimeo_id, "split": r.split, "file_name": r.file_name}
            for r in mp.itertuples(index=False)
        }

    def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self.runs_dir.glob("*.trec"))

    def list_queries(self, run_name: str) -> list[str]:
        run = load_run_trec(self.runs_dir / f"{run_name}.trec")
        return sorted(run.keys())

    def _qrels_for(self, run_name: str) -> dict:
        # Heurística: el split va al final del run name (…_valid / _test / _train).
        for split in ("valid", "test", "train"):
            if run_name.endswith(split):
                path = self.qrels_dir / f"{split}.qrels"
                if path.exists():
                    return load_qrels_trec(path)
        return {}

    def search(self, run_name: str, query_id: str, top_k: int = 20) -> list[dict]:
        run = load_run_trec(self.runs_dir / f"{run_name}.trec")
        ranked = sorted(run.get(query_id, {}).items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        qrels = self._qrels_for(run_name).get(query_id, {})

        results = []
        for rank, (image_id, score) in enumerate(ranked, start=1):
            meta = self._map.get(image_id, {})
            vimeo_id = meta.get("vimeo_id")
            path = None
            if meta:
                path = str(self.raw_root / meta["split"] / meta["file_name"])
            results.append({
                "rank": rank,
                "image_id": image_id,
                "score": round(float(score), 4),
                "vimeo_id": vimeo_id,
                "vimeo_url": f"https://vimeo.com/{vimeo_id}" if vimeo_id else None,
                "image_path": path,
                "relevant": bool(qrels.get(image_id, 0)),
            })
        return results
