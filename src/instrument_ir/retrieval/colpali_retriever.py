"""Embedder ColQwen (B3 late-interaction). Requiere el extra `[colpali]` y GPU.

Usa `colpali-engine` (illuin-tech) para producir embeddings multivector de imágenes y queries; el
scoring MaxSim lo hace `LateInteractionRetriever`. Imports perezosos: solo se cargan al instanciar.
"""

from __future__ import annotations

import numpy as np

from ..utils.io import ImageProvider
from .cache import MultiVectorCache
from .late_interaction import LateInteractionRetriever


class ColQwenEmbedder:
    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str | None = None,
        batch_size: int = 8,
    ):
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.model_id = f"colqwen_{model_name.split('/')[-1]}"
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.model = ColQwen2.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=self.device
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)

    def _to_list(self, batch_tensor) -> list[np.ndarray]:
        # batch_tensor: [B, n_tokens, d] -> lista de [n_tokens, d] en float32 numpy
        return [t.to(self._torch.float32).cpu().numpy() for t in batch_tensor]

    def encode_images(self, image_ids: list[str], provider: ImageProvider) -> list[np.ndarray]:
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(image_ids), self.batch_size):
            imgs = [provider.load(i) for i in image_ids[start : start + self.batch_size]]
            batch = self.processor.process_images(imgs).to(self.device)
            with torch.no_grad():
                emb = self.model(**batch)
            out.extend(self._to_list(emb))
        return out

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.processor.process_queries(
                texts[start : start + self.batch_size]
            ).to(self.device)
            with torch.no_grad():
                emb = self.model(**batch)
            out.extend(self._to_list(emb))
        return out


class ColQwenRetriever(LateInteractionRetriever):
    def __init__(
        self,
        image_provider: ImageProvider,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str | None = None,
        batch_size: int = 8,
        cache: MultiVectorCache | None = None,
        split: str | None = None,
    ):
        embedder = ColQwenEmbedder(model_name, device, batch_size)
        super().__init__(embedder, image_provider, cache=cache, split=split)
