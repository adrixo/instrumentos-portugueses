"""Orquestación de un reranking pointwise (B4): candidatos -> VLM -> artefactos.

Produce (ADR §4):
  outputs/candidates/{run_id}.parquet     # query_id, image_id, dense_rank, dense_score
  outputs/rerank_traces/{run_id}.jsonl    # una traza por candidato
  outputs/runs/{run_id}.trec              # runfile reordenado (final_top_k)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..data.queries import Query
from ..retrieval.base import ScoredDoc
from ..utils.trec import write_run_trec
from .base import Candidate, Reranker


def run_pointwise_rerank(
    reranker: Reranker,
    queries: list[Query],
    instruments: dict,
    candidates_by_query: dict[str, list[Candidate]],
    run_id: str,
    final_top_k: int = 100,
    outputs_root: Path = Path("outputs"),
) -> Path:
    outputs_root = Path(outputs_root)
    traces_path = outputs_root / "rerank_traces" / f"{run_id}.jsonl"
    cand_path = outputs_root / "candidates" / f"{run_id}.parquet"
    run_path = outputs_root / "runs" / f"{run_id}.trec"
    for p in (traces_path, cand_path, run_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    qmap = {q.query_id: q for q in queries}
    rankings: dict[str, list[ScoredDoc]] = {}
    cand_rows: list[dict] = []

    with traces_path.open("w", encoding="utf-8") as tf:
        for query_id, cands in candidates_by_query.items():
            query = qmap.get(query_id)
            if query is None:
                continue
            instrument = instruments.get(query.instrument_id, {"canonical_name": query.instrument_id})
            for c in cands:
                cand_rows.append(
                    {"query_id": query_id, "image_id": c.image_id,
                     "dense_rank": c.dense_rank, "dense_score": c.dense_score, "run_id": run_id}
                )
            reranked, traces = reranker.rerank(query, instrument, cands, run_id)
            for tr in traces:
                tf.write(json.dumps(tr, ensure_ascii=False) + "\n")
            rankings[query_id] = [
                ScoredDoc(d.image_id, d.final_score) for d in reranked[:final_top_k]
            ]

    pd.DataFrame(cand_rows).to_parquet(cand_path, index=False)
    write_run_trec(rankings, run_id, run_path)
    return run_path
