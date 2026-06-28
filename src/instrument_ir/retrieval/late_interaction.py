"""Retrieval late-interaction / multivector (B3, ADR §4).

A diferencia de B1 (un vector por imagen), aquí cada imagen es un conjunto de vectores (patches) y la
query un conjunto de vectores (tokens). El score es MaxSim: por cada token de la query se toma el
máximo producto con los patches de la imagen y se suman.

El scoring (`maxsim`) es una función pura y testeable sin GPU. El embedder concreto (ColQwen) se
inyecta; para tests se usa `MockMultiVectorEmbedder`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..data.queries import Query
from ..utils.io import ImageProvider
from .base import BaseRetriever, ScoredDoc


def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """MaxSim: sum_t max_p (q_t · d_p). query_vecs [nq, d], doc_vecs [nd, d]."""
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return 0.0
    sim = query_vecs @ doc_vecs.T  # [nq, nd]
    return float(sim.max(axis=1).sum())


class MultiVectorEmbedder(Protocol):
    model_id: str

    def encode_images(self, image_ids: list[str], provider: ImageProvider) -> list[np.ndarray]: ...
    def encode_queries(self, texts: list[str]) -> list[np.ndarray]: ...


class LateInteractionRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: MultiVectorEmbedder,
        provider: ImageProvider,
        cache=None,
        split: str | None = None,
    ):
        self.embedder = embedder
        self.provider = provider
        self.cache = cache
        self.split = split
        self.name = embedder.model_id.replace("/", "-").lower()

    def encode_corpus(self, image_ids: list[str]) -> list[np.ndarray]:
        if self.cache and self.split:
            cached = self.cache.load(self.embedder.model_id, self.split)
            if cached is not None:
                cached_ids, vecs = cached
                pos = {iid: i for i, iid in enumerate(cached_ids)}
                if all(iid in pos for iid in image_ids):
                    return [vecs[pos[iid]] for iid in image_ids]
        vecs = self.embedder.encode_images(image_ids, self.provider)
        if self.cache and self.split:
            self.cache.save(self.embedder.model_id, self.split, image_ids, vecs)
        return vecs

    def rank(
        self, queries: list[Query], image_ids: list[str], top_k: int
    ) -> dict[str, list[ScoredDoc]]:
        corpus = self.encode_corpus(image_ids)
        q_vecs = self.embedder.encode_queries([q.text for q in queries])

        out: dict[str, list[ScoredDoc]] = {}
        for qi, q in enumerate(queries):
            scored = [
                ScoredDoc(image_ids[di], maxsim(q_vecs[qi], corpus[di]))
                for di in range(len(image_ids))
            ]
            scored.sort(key=lambda d: d.score, reverse=True)
            out[q.query_id] = scored[:top_k]
        return out


class MockMultiVectorEmbedder:
    """Embedder multivector determinista para tests (sin modelo). Mapea ids/textos a vectores fijos."""

    model_id = "mock_multivector"

    def __init__(self, image_vecs: dict[str, np.ndarray], text_vecs: dict[str, np.ndarray]):
        self._img = image_vecs
        self._txt = text_vecs

    def encode_images(self, image_ids: list[str], provider: ImageProvider) -> list[np.ndarray]:
        return [self._img[i].astype(np.float32) for i in image_ids]

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        return [self._txt[t].astype(np.float32) for t in texts]
