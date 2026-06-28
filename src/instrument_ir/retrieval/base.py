"""Interfaz común de retrieval (ADR §9: retrieval/base.py).

Tanto el dense global (B1) como el late-interaction (B3) implementan `BaseRetriever`. La salida común
es un ranking por query, que luego se exporta a runfile TREC. Las representaciones internas
(vector global vs multivector) son privadas de cada implementación.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..data.queries import Query


@dataclass(frozen=True)
class ScoredDoc:
    image_id: str
    score: float


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def rank(
        self, queries: list[Query], image_ids: list[str], top_k: int
    ) -> dict[str, list[ScoredDoc]]:
        """Devuelve {query_id: [ScoredDoc ordenado desc por score]} truncado a top_k."""
        raise NotImplementedError


class DummyRetriever(BaseRetriever):
    """Retriever determinista sin modelo (Fase 1 / smoke, ADR §27).

    Asigna a cada par (query, imagen) un score pseudo-aleatorio pero reproducible (hash). No tiene
    valor semántico; solo sirve para validar el tubo (runfile + evaluación) sin dependencias pesadas.
    """

    name = "dummy"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _score(self, query_id: str, image_id: str) -> float:
        h = hashlib.sha256(f"{self.seed}:{query_id}:{image_id}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def rank(
        self, queries: list[Query], image_ids: list[str], top_k: int
    ) -> dict[str, list[ScoredDoc]]:
        out: dict[str, list[ScoredDoc]] = {}
        for q in queries:
            scored = [ScoredDoc(iid, self._score(q.query_id, iid)) for iid in image_ids]
            scored.sort(key=lambda d: d.score, reverse=True)
            out[q.query_id] = scored[:top_k]
        return out
