"""Retriever dense global con OpenCLIP (B1, ADR §4). Requiere el extra `[dense]`.

Codifica imágenes y texto al mismo espacio y ordena por coseno (índice flat exacto). No se usa en la
Fase 1 (smoke con DummyRetriever); entra en la Fase 2.
"""

from __future__ import annotations

import numpy as np

from ..data.queries import Query
from ..utils.io import ImageProvider
from .base import BaseRetriever, ScoredDoc
from .faiss_index import FlatIPIndex


class OpenClipRetriever(BaseRetriever):
    def __init__(
        self,
        image_provider: ImageProvider,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        batch_size: int = 64,
    ):
        import open_clip
        import torch

        self.name = f"openclip_{model_name}_{pretrained}".lower().replace("/", "-")
        self.image_provider = image_provider
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def _encode_images(self, image_ids: list[str]) -> np.ndarray:
        torch = self._torch
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(image_ids), self.batch_size):
                batch_ids = image_ids[start : start + self.batch_size]
                imgs = torch.stack(
                    [self.preprocess(self.image_provider.load(i)) for i in batch_ids]
                ).to(self.device)
                emb = self.model.encode_image(imgs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append(emb.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def rank(
        self, queries: list[Query], image_ids: list[str], top_k: int
    ) -> dict[str, list[ScoredDoc]]:
        corpus = self._encode_images(image_ids)
        index = FlatIPIndex(dim=corpus.shape[1])
        index.add(corpus)

        q_emb = self._encode_texts([q.text for q in queries])
        scores, idx = index.search(q_emb, min(top_k, len(image_ids)))

        out: dict[str, list[ScoredDoc]] = {}
        for qi, q in enumerate(queries):
            out[q.query_id] = [
                ScoredDoc(image_ids[int(idx[qi, r])], float(scores[qi, r]))
                for r in range(idx.shape[1])
            ]
        return out
