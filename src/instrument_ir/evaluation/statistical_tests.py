"""Análisis estadístico de comparación entre sistemas (ADR §6.4).

- bootstrap_ci: intervalo de confianza por remuestreo de queries.
- paired_permutation_test: test pareado (randomization) sobre las mismas queries.
- holm_bonferroni: corrección por comparaciones múltiples.

Reproducible: todo depende del seed. Unidad de remuestreo = query (ADR: ~22 clases × idiomas → CIs anchos).
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 42
) -> dict:
    """IC del (1-alpha) para la media, por remuestreo con reemplazo."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi)}


def paired_permutation_test(
    a: list[float], b: list[float], n_perm: int = 10000, seed: int = 42
) -> float:
    """p-valor (dos colas) del test de permutación pareado para H0: media(a-b)=0."""
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    assert da.shape == db.shape, "a y b deben estar pareados (mismas queries)"
    diff = da - db
    if diff.size == 0:
        return 1.0
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_perm, diff.size))
    perm_means = np.abs((signs * diff).mean(axis=1))
    # +1 (regla de continuidad) para no devolver p=0.
    return float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Corrección Holm-Bonferroni. Devuelve {nombre: {p, p_adj, significant}}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev_adj = 0.0
    for i, (name, p) in enumerate(items):
        p_adj = min(1.0, max(prev_adj, (m - i) * p))
        prev_adj = p_adj
        out[name] = {"p": p, "p_adj": p_adj, "significant": p_adj < alpha}
    return out


def compare_systems(
    per_query_a: dict[str, float], per_query_b: dict[str, float], seed: int = 42
) -> dict:
    """Compara dos sistemas en una métrica (per-query dicts alineados por query_id)."""
    common = sorted(set(per_query_a) & set(per_query_b))
    a = [per_query_a[q] for q in common]
    b = [per_query_b[q] for q in common]
    diff = [x - y for x, y in zip(a, b)]
    return {
        "n": len(common),
        "mean_a": float(np.mean(a)) if a else 0.0,
        "mean_b": float(np.mean(b)) if b else 0.0,
        "delta_ci": bootstrap_ci(diff, seed=seed),
        "p_value": paired_permutation_test(a, b, seed=seed),
    }
