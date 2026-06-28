"""Contratos del reranking (B4/B5, ADR §4-§5).

Un Reranker recibe los candidatos (top-N) de una etapa dense y los reordena inspeccionando la imagen.
B4 (VLM pointwise) y B5 (agente) comparten esta interfaz; el llamador no los distingue.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..data.queries import Query


@dataclass(frozen=True)
class Candidate:
    image_id: str
    dense_rank: int
    dense_score: float


@dataclass(frozen=True)
class RerankedDoc:
    image_id: str
    final_score: float
    final_rank: int
    dense_rank: int
    dense_score: float
    decision: str
    confidence: float


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(
        self, query: Query, instrument: dict, candidates: list[Candidate], run_id: str
    ) -> tuple[list[RerankedDoc], list[dict]]:
        """Devuelve (docs reordenados con final_rank, trazas estructuradas por candidato)."""
        raise NotImplementedError


def load_candidates_from_run(run_path, top_n: int) -> dict[str, list[Candidate]]:
    """Lee un runfile dense (B1/B3) y toma los top-N candidatos por query."""
    from ..utils.trec import load_run_trec

    run = load_run_trec(run_path)
    out: dict[str, list[Candidate]] = {}
    for query_id, docs in run.items():
        ordered = sorted(docs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        out[query_id] = [
            Candidate(image_id=iid, dense_rank=rank, dense_score=score)
            for rank, (iid, score) in enumerate(ordered, start=1)
        ]
    return out
