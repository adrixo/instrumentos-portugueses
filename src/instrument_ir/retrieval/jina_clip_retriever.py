"""Embedder JinaCLIP v2 (B1, modelo multimodal multilingüe). Requiere el extra `[dense]`.

JinaCLIP v2 es multilingüe → relevante para las queries pt/en/es (a diferencia de OpenCLIP, inglés).
Usa `transformers` con `trust_remote_code=True`.
"""

from __future__ import annotations

import numpy as np

from ..utils.io import ImageProvider
from .cache import CorpusEmbeddingCache
from .dense import DenseRetriever


class JinaClipEmbedder:
    def __init__(
        self,
        model_name: str = "jinaai/jina-clip-v2",
        device: str | None = None,
        batch_size: int = 32,
        truncate_dim: int | None = None,
    ):
        import torch
        from transformers import AutoModel

        self.model_id = f"jinaclip_{model_name.split('/')[-1]}"
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device).eval()

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        emb = emb.astype(np.float32)
        norms = np.linalg.norm(emb, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def encode_images(self, image_ids: list[str], provider: ImageProvider) -> np.ndarray:
        feats: list[np.ndarray] = []
        for start in range(0, len(image_ids), self.batch_size):
            batch = [provider.load(i) for i in image_ids[start : start + self.batch_size]]
            emb = self.model.encode_image(
                batch, truncate_dim=self.truncate_dim
            )
            feats.append(np.asarray(emb, dtype=np.float32))
        return self._normalize(np.concatenate(feats, axis=0))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode_text(texts, truncate_dim=self.truncate_dim)
        return self._normalize(np.asarray(emb, dtype=np.float32))


class JinaClipRetriever(DenseRetriever):
    def __init__(
        self,
        image_provider: ImageProvider,
        model_name: str = "jinaai/jina-clip-v2",
        device: str | None = None,
        batch_size: int = 32,
        truncate_dim: int | None = None,
        cache: CorpusEmbeddingCache | None = None,
        split: str | None = None,
    ):
        embedder = JinaClipEmbedder(model_name, device, batch_size, truncate_dim)
        super().__init__(embedder, image_provider, cache=cache, split=split)
