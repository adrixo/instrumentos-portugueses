"""Embedder OpenCLIP (B1, ADR §4). Requiere el extra `[dense]`.

`OpenClipEmbedder` codifica imágenes/texto a vectores L2-normalizados; el ranking lo hace
`DenseRetriever`. `OpenClipRetriever` es un atajo de conveniencia (compatibilidad).
"""

from __future__ import annotations

import numpy as np

from ..utils.io import ImageProvider
from .cache import CorpusEmbeddingCache
from .dense import DenseRetriever


class OpenClipEmbedder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        batch_size: int = 64,
    ):
        import open_clip
        import torch

        self.model_id = f"openclip_{model_name}_{pretrained}"
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_images(self, image_ids: list[str], provider: ImageProvider) -> np.ndarray:
        torch = self._torch
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(image_ids), self.batch_size):
                batch_ids = image_ids[start : start + self.batch_size]
                imgs = torch.stack(
                    [self.preprocess(provider.load(i)) for i in batch_ids]
                ).to(self.device)
                emb = self.model.encode_image(imgs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(feats, axis=0)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32)


class OpenClipRetriever(DenseRetriever):
    def __init__(
        self,
        image_provider: ImageProvider,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        batch_size: int = 64,
        cache: CorpusEmbeddingCache | None = None,
        split: str | None = None,
    ):
        embedder = OpenClipEmbedder(model_name, pretrained, device, batch_size)
        super().__init__(embedder, image_provider, cache=cache, split=split)
