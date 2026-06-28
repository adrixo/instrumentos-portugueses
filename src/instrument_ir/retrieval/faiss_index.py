"""Índice exacto por producto interno (ADR: FAISS, índice no especificado → flat exacto).

A ~1.3k vectores un índice ANN no aporta velocidad y mete error de aproximación; se usa
`faiss.IndexFlatIP` (exacto) si faiss está disponible, con fallback a numpy puro.
"""

from __future__ import annotations

import numpy as np


class FlatIPIndex:
    """Búsqueda exacta por producto interno. Asume vectores ya normalizados (→ coseno)."""

    def __init__(self, dim: int):
        import os

        self.dim = dim
        self._use_faiss = False
        self._matrix: np.ndarray | None = None
        # INSTRUMENT_IR_NO_FAISS=1 fuerza el camino numpy (evita el clash OpenMP faiss+torch en macOS).
        if os.environ.get("INSTRUMENT_IR_NO_FAISS") == "1":
            return
        try:
            import faiss

            self._index = faiss.IndexFlatIP(dim)
            self._use_faiss = True
        except ImportError:
            pass

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        if self._use_faiss:
            self._index.add(vectors)
        else:
            self._matrix = vectors

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (scores, indices) con shape [n_queries, top_k]."""
        queries = np.ascontiguousarray(queries.astype(np.float32))
        if self._use_faiss:
            return self._index.search(queries, top_k)
        sims = queries @ self._matrix.T  # type: ignore[union-attr]
        idx = np.argsort(-sims, axis=1)[:, :top_k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores, idx
