"""Métricas específicas de reranking (ADR §6.2).

- candidate_recall@N: cuántos relevantes están en el top-N del dense (entrada al reranker).
- oracle_recall@K: techo de recall@K tras rerank (no se puede recuperar lo que no entró en top-N).
- rerank_gain@K: recall@K(reranked) - recall@K(dense).
- delta_ndcg@K / delta_map: mejora del reranker sobre el dense.

oracle_recall es crítico: si un positivo no entró en el top-N inicial, el reranker no puede recuperarlo.
"""

from __future__ import annotations


def _recall_at_k(ranked_ids: list[str], rel: set[str], k: int) -> float:
    if not rel:
        return 0.0
    topk = ranked_ids[:k]
    return len(rel.intersection(topk)) / len(rel)


def _ranked_ids(run_q: dict[str, float]) -> list[str]:
    return [iid for iid, _ in sorted(run_q.items(), key=lambda kv: kv[1], reverse=True)]


def candidate_recall_at_n(dense_run: dict, qrels: dict, n: int = 200) -> dict:
    """Por query y macro: fracción de relevantes presentes en el top-N de candidatos."""
    per_q = {}
    for qid, rel_map in qrels.items():
        rel = {d for d, r in rel_map.items() if r > 0}
        cand = _ranked_ids(dense_run.get(qid, {}))[:n]
        per_q[qid] = (len(rel.intersection(cand)) / len(rel)) if rel else 0.0
    macro = sum(per_q.values()) / len(per_q) if per_q else 0.0
    return {"per_query": per_q, "macro": macro}


def oracle_recall_at_k(dense_run: dict, qrels: dict, n: int = 200, k: int = 100) -> dict:
    """Techo de recall@K alcanzable tras rerankear el top-N (relevantes en top-N, recolocados arriba)."""
    per_q = {}
    for qid, rel_map in qrels.items():
        rel = {d for d, r in rel_map.items() if r > 0}
        if not rel:
            per_q[qid] = 0.0
            continue
        cand = set(_ranked_ids(dense_run.get(qid, {}))[:n])
        rel_in_cand = len(rel.intersection(cand))
        per_q[qid] = min(rel_in_cand, k) / len(rel)
    macro = sum(per_q.values()) / len(per_q) if per_q else 0.0
    return {"per_query": per_q, "macro": macro}


def rerank_gain_at_k(reranked_run: dict, dense_run: dict, qrels: dict, k: int = 100) -> dict:
    """recall@K(reranked) - recall@K(dense), por query y macro."""
    per_q = {}
    for qid, rel_map in qrels.items():
        rel = {d for d, r in rel_map.items() if r > 0}
        r_re = _recall_at_k(_ranked_ids(reranked_run.get(qid, {})), rel, k)
        r_de = _recall_at_k(_ranked_ids(dense_run.get(qid, {})), rel, k)
        per_q[qid] = r_re - r_de
    macro = sum(per_q.values()) / len(per_q) if per_q else 0.0
    return {"per_query": per_q, "macro": macro}


def delta_metric(reranked_run: dict, dense_run: dict, qrels: dict, metric: str = "ndcg@100") -> float:
    """metric(reranked) - metric(dense) usando ranx (agregado)."""
    from ranx import Qrels, Run, evaluate

    common = sorted(set(qrels) & set(reranked_run) & set(dense_run))
    q = Qrels({c: qrels[c] for c in common})
    re = evaluate(q, Run({c: reranked_run[c] for c in common}), metric)
    de = evaluate(q, Run({c: dense_run[c] for c in common}), metric)
    return float(re) - float(de)
