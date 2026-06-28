"""Factory de retrievers por nombre/config (ADR §4: backends intercambiables).

Resuelve `retriever.type` desde un dict de config a una instancia. Mantiene el core desacoplado de los
modelos concretos (lazy import de torch dentro de cada retriever).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..utils.io import ImageProvider
from .base import BaseRetriever, DummyRetriever
from .cache import CorpusEmbeddingCache

# Atajos -> config de modelo, para usar `--model nombre` sin un YAML.
SHORTHANDS: dict[str, dict] = {
    "dummy": {"type": "dummy"},
    "openclip-vitb32": {"type": "openclip", "model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
    "openclip-vitl14": {"type": "openclip", "model_name": "ViT-L-14", "pretrained": "laion2b_s34b_b79k"},
    "jinaclip": {"type": "jinaclip", "model_name": "jinaai/jina-clip-v2"},
}


def load_model_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg.get("retriever", cfg)  # admite {retriever: {...}} o el dict directo


def build_retriever(
    cfg: dict,
    provider: ImageProvider | None = None,
    cache: CorpusEmbeddingCache | None = None,
    split: str | None = None,
) -> BaseRetriever:
    rtype = cfg["type"]

    if rtype == "dummy":
        return DummyRetriever(seed=cfg.get("seed", 42))

    if provider is None:
        raise ValueError(f"El retriever '{rtype}' necesita un ImageProvider")

    if rtype == "openclip":
        from .openclip_retriever import OpenClipRetriever

        return OpenClipRetriever(
            provider,
            model_name=cfg.get("model_name", "ViT-B-32"),
            pretrained=cfg.get("pretrained", "laion2b_s34b_b79k"),
            batch_size=cfg.get("batch_size", 64),
            cache=cache,
            split=split,
        )

    if rtype == "jinaclip":
        from .jina_clip_retriever import JinaClipRetriever

        return JinaClipRetriever(
            provider,
            model_name=cfg.get("model_name", "jinaai/jina-clip-v2"),
            batch_size=cfg.get("batch_size", 32),
            truncate_dim=cfg.get("truncate_dim"),
            cache=cache,
            split=split,
        )

    raise ValueError(f"Tipo de retriever desconocido: {rtype}")


def resolve_model(model: str) -> dict:
    """Resuelve un `--model`: atajo conocido o ruta a un YAML de configs/models/."""
    if model in SHORTHANDS:
        return dict(SHORTHANDS[model])
    path = Path(model)
    if path.exists():
        return load_model_config(path)
    raise ValueError(f"Modelo no reconocido: {model}. Atajos: {list(SHORTHANDS)} o ruta YAML.")
