"""Normalización del score del VLM (ADR §4 B4).

Regla por defecto:
    present   -> confidence
    uncertain -> 0.5 * confidence
    absent    -> 0.0
El desempate (tie-breaker) es el dense_score, aplicado al ordenar.
"""

from __future__ import annotations


def normalize_score(decision: str, confidence: float) -> float:
    confidence = max(0.0, min(1.0, float(confidence)))
    if decision == "present":
        return confidence
    if decision == "uncertain":
        return 0.5 * confidence
    return 0.0
