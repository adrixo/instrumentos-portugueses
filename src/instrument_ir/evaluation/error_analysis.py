"""Error analysis: falsos positivos y negativos por query (ADR §12.3, §13).

Para un runfile y qrels, identifica en el top-K:
- falsos positivos: recuperados arriba que NO contienen el instrumento.
- falsos negativos: relevantes que NO aparecen en el top-K.
Resuelve vimeo_id vía el mapping (solo para inspección humana, no inferencia).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.qrels import load_qrels_trec
from ..utils.trec import load_run_trec


def analyze_run(
    run_path: Path, qrels_path: Path, mapping_path: Path, k: int = 10, max_examples: int = 20
) -> dict:
    run = load_run_trec(run_path)
    qrels = load_qrels_trec(qrels_path)
    mp = pd.read_parquet(mapping_path)
    vimeo = {r.image_id: r.vimeo_id for r in mp.itertuples(index=False)}

    per_query: dict[str, dict] = {}
    for qid, rel_map in qrels.items():
        rel = {d for d, r in rel_map.items() if r > 0}
        ranked = [iid for iid, _ in sorted(run.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True)]
        topk = ranked[:k]
        fps = [iid for iid in topk if iid not in rel]
        fns = [iid for iid in rel if iid not in set(topk)]
        per_query[qid] = {
            "false_positives": [{"image_id": i, "vimeo_id": vimeo.get(i)} for i in fps[:max_examples]],
            "false_negatives": [{"image_id": i, "vimeo_id": vimeo.get(i)} for i in fns[:max_examples]],
            "n_fp": len(fps),
            "n_fn": len(fns),
        }
    return per_query
