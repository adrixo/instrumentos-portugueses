"""Evaluación IR con ranx (ADR §6, §8.4 stage `evaluate`).

Calcula métricas agregadas, por query y por instrumento (macro), y vuelca un JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..data.qrels import load_qrels_trec
from ..data.queries import Query
from ..utils.trec import load_run_trec

DEFAULT_METRICS = (
    "recall@10",
    "recall@20",
    "recall@50",
    "recall@100",
    "precision@10",
    "ndcg@10",
    "ndcg@20",
    "ndcg@100",
    "map",
    "mrr",
)


def evaluate_run(
    run_path: Path,
    qrels_path: Path,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    queries: list[Query] | None = None,
    out_path: Path | None = None,
) -> dict:
    from ranx import Qrels, Run, evaluate

    qrels_dict = load_qrels_trec(qrels_path)
    run_dict = load_run_trec(run_path)

    # ranx exige que las queries del run estén en qrels; nos quedamos con la intersección.
    common = sorted(set(qrels_dict) & set(run_dict))
    qrels = Qrels({q: qrels_dict[q] for q in common})
    run = Run({q: run_dict[q] for q in common})

    metric_list = list(metrics)
    aggregate = evaluate(qrels, run, metric_list)
    per_query = evaluate(qrels, run, metric_list, return_mean=False)

    # per_query de ranx: {metric: np.ndarray alineado con qrels.keys()}
    qids = list(qrels.qrels.keys())
    per_query_out: dict[str, dict[str, float]] = {}
    for mi, metric in enumerate(metric_list):
        arr = per_query[metric] if len(metric_list) > 1 else per_query
        for qi, qid in enumerate(qids):
            per_query_out.setdefault(qid, {})[metric] = float(arr[qi])

    result = {
        "run": str(run_path),
        "qrels": str(qrels_path),
        "n_queries": len(common),
        "metrics": {m: float(aggregate[m]) for m in metric_list}
        if len(metric_list) > 1
        else {metric_list[0]: float(aggregate)},
        "per_query": per_query_out,
    }

    # Macro por instrumento (promedio de queries del mismo instrumento).
    if queries is not None:
        qid_to_instrument = {q.query_id: q.instrument_id for q in queries}
        by_instrument: dict[str, list[dict[str, float]]] = {}
        for qid, vals in per_query_out.items():
            instrument = qid_to_instrument.get(qid)
            if instrument is not None:
                by_instrument.setdefault(instrument, []).append(vals)
        per_instrument: dict[str, dict[str, float]] = {}
        for instrument, rows in by_instrument.items():
            per_instrument[instrument] = {
                m: sum(r[m] for r in rows) / len(rows) for m in metric_list
            }
        result["per_instrument"] = per_instrument
        if per_instrument:
            result["macro_metrics"] = {
                m: sum(pi[m] for pi in per_instrument.values()) / len(per_instrument)
                for m in metric_list
            }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result
