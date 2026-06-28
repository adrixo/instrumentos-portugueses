"""Retriever dense genérico (B1): embedder intercambiable + caché + índice flat exacto.

Cualquier modelo dense global (OpenCLIP, JinaCLIP, …) implementa el protocolo `Embedder`:
codifica imágenes y texto a vectores L2-normalizados en el mismo espacio. `DenseRetriever` se encarga
del cacheo de corpus, el índice y el ranking, igual para todos los modelos.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..data.queries import Query
from ..utils.io import ImageProvider
from .base import BaseRetriever, ScoredDoc
from .cache import CorpusEmbeddingCache
from .faiss_index import FlatIPIndex


class Embedder(Protocol):
    model_id: str

    def encode_images(self, image_ids: list[str], provider: ImageProvider) -> np.ndarray: ...
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: Embedder,
        provider: ImageProvider,
        cache: CorpusEmbeddingCache | None = None,
        split: str | None = None,
    ):
        self.embedder = embedder
        self.provider = provider
        self.cache = cache
        self.split = split
        self.name = embedder.model_id.replace("/", "-").lower()

    def encode_corpus(self, image_ids: list[str]) -> np.ndarray:
        """Codifica (o recupera de caché) los embeddings del corpus alineados con image_ids."""
        if self.cache and self.split:
            cached = self.cache.load(self.embedder.model_id, self.split)
            if cached is not None:
                cached_ids, cached_emb = cached
                pos = {iid: i for i, iid in enumerate(cached_ids)}
                if all(iid in pos for iid in image_ids):
                    return cached_emb[[pos[iid] for iid in image_ids]]

        emb = self.embedder.encode_images(image_ids, self.provider)
        if self.cache and self.split:
            self.cache.save(self.embedder.model_id, self.split, image_ids, emb)
        return emb

    def rank(
        self, queries: list[Query], image_ids: list[str], top_k: int
    ) -> dict[str, list[ScoredDoc]]:
        corpus = self.encode_corpus(image_ids)
        index = FlatIPIndex(dim=corpus.shape[1])
        index.add(corpus)

        q_emb = self.embedder.encode_texts([q.text for q in queries])
        scores, idx = index.search(q_emb, min(top_k, len(image_ids)))

        out: dict[str, list[ScoredDoc]] = {}
        for qi, q in enumerate(queries):
            out[q.query_id] = [
                ScoredDoc(image_ids[int(idx[qi, r])], float(scores[qi, r]))
                for r in range(idx.shape[1])
            ]
        return out
