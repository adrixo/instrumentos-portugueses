"""Caché de embeddings de corpus para no recodificar entre experimentos (ADR §6.3 coste).

Clave = (model_id, split). Se guarda como .npz con los image_id y la matriz de embeddings
(ya L2-normalizada). Si el modelo cambia, cambia el model_id → la caché se invalida sola.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CorpusEmbeddingCache:
    def __init__(self, root: Path = Path("outputs/cache/embeddings")):
        self.root = Path(root)

    def _path(self, model_id: str, split: str) -> Path:
        safe = model_id.replace("/", "-")
        return self.root / safe / f"{split}.npz"

    def load(self, model_id: str, split: str) -> tuple[list[str], np.ndarray] | None:
        path = self._path(model_id, split)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        return list(data["image_ids"]), data["embeddings"]

    def save(self, model_id: str, split: str, image_ids: list[str], embeddings: np.ndarray) -> Path:
        path = self._path(model_id, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, image_ids=np.array(image_ids, dtype=object), embeddings=embeddings)
        return path


class MultiVectorCache:
    """Caché para embeddings multivector (B3): por imagen una matriz [n_patches, d] de tamaño variable."""

    def __init__(self, root: Path = Path("outputs/cache/multivector")):
        self.root = Path(root)

    def _path(self, model_id: str, split: str) -> Path:
        safe = model_id.replace("/", "-")
        return self.root / safe / f"{split}.npz"

    def load(self, model_id: str, split: str) -> tuple[list[str], list[np.ndarray]] | None:
        path = self._path(model_id, split)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        return list(data["image_ids"]), list(data["embeddings"])

    def save(
        self, model_id: str, split: str, image_ids: list[str], embeddings: list[np.ndarray]
    ) -> Path:
        path = self._path(model_id, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            image_ids=np.array(image_ids, dtype=object),
            embeddings=np.array(embeddings, dtype=object),
        )
        return path
