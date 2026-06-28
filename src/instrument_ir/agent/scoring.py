"""Fusión de evidencias del agente B5 (ADR §5). Funciones puras.

Versión simple (preferida para el paper):  final = max(score_full, score_crop)
Versión opcional (ponderada):              0.6*max(full,crop) + 0.2*caption + 0.2*dense_norm
"""

from __future__ import annotations


def fuse_max(score_full: float, score_crop: float) -> float:
    return max(score_full, score_crop)


def fuse_weighted(
    score_full: float, score_crop: float, score_caption: float, dense_norm: float
) -> float:
    return 0.60 * max(score_full, score_crop) + 0.20 * score_caption + 0.20 * dense_norm


def fuse(strategy: str, *, score_full: float, score_crop: float = 0.0,
         score_caption: float = 0.0, dense_norm: float = 0.0) -> float:
    if strategy == "weighted":
        return fuse_weighted(score_full, score_crop, score_caption, dense_norm)
    return fuse_max(score_full, score_crop)
