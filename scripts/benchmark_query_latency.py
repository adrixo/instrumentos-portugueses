#!/usr/bin/env python3
"""Benchmark per-query retrieval/reranking latency for slide/report figures.

Dense and late-interaction systems are timed by loading each model once and ranking
the selected queries one by one. VLM/agentic rerankers can be summarized from trace
timestamps produced by completed runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from instrument_ir.data.prepare_dataset import resolve_mapping
from instrument_ir.data.queries import load_queries
from instrument_ir.retrieval.factory import build_retriever, resolve_model
from instrument_ir.utils.io import ImageProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure query latency for IR systems.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/portuguese_instruments"))
    parser.add_argument("--queries", type=Path, default=Path("configs/queries.yaml"))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--models",
        default="openclip-vitb32,openclip-vitl14,jinaclip,colqwen",
        help="Comma-separated retriever models to benchmark. Use empty string to skip.",
    )
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        help="Trace JSONL to summarize as label=path, e.g. B4=outputs/rerank_traces/B4_test.jsonl",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/metrics/query_latency.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/metrics/query_latency_summary.json"))
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize(rows: list[dict]) -> dict:
    out = {}
    by_system: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(float(row["latency_seconds"]))
    for system, values in by_system.items():
        out[system] = {
            "n_queries": len(values),
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "p25_seconds": percentile(values, 0.25),
            "p75_seconds": percentile(values, 0.75),
            "p95_seconds": percentile(values, 0.95),
        }
    return out


def benchmark_retrievers(args: argparse.Namespace) -> list[dict]:
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        return []

    mapping = resolve_mapping(args.processed, args.split)
    provider = ImageProvider(mapping, args.raw)
    image_ids = pd.read_parquet(args.processed / args.split / "corpus.parquet")[
        "image_id"
    ].tolist()
    queries = load_queries(args.queries)

    rows = []
    for model_name in model_names:
        cfg = resolve_model(model_name)
        retriever = build_retriever(cfg, provider=provider, split=args.split)
        for query in queries:
            start = time.perf_counter()
            retriever.rank([query], image_ids, args.top_k)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "system": model_name,
                    "query_id": query.query_id,
                    "instrument": query.instrument_id,
                    "latency_seconds": f"{elapsed:.6f}",
                    "source": "benchmark",
                    "top_k": args.top_k,
                }
            )
            print(f"{model_name}\t{query.query_id}\t{elapsed:.3f}s", flush=True)
    return rows


def parse_trace_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Trace spec must be label=path: {spec}")
        label, path = spec.split("=", 1)
        out.append((label.strip(), Path(path)))
    return out


def summarize_trace(label: str, path: Path) -> list[dict]:
    if not path.exists():
        return []
    stamps: dict[str, list[datetime]] = defaultdict(list)
    instruments: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        query_id = item.get("query_id")
        stamp = item.get("timestamp_utc")
        if not query_id or not stamp:
            continue
        stamps[query_id].append(datetime.fromisoformat(stamp.replace("Z", "+00:00")))
        instruments[query_id] = item.get("instrument", "")

    rows = []
    for query_id, values in stamps.items():
        if len(values) < 2:
            continue
        elapsed = (max(values) - min(values)).total_seconds()
        rows.append(
            {
                "system": label,
                "query_id": query_id,
                "instrument": instruments.get(query_id, ""),
                "latency_seconds": f"{elapsed:.6f}",
                "source": "trace_span",
                "top_k": "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = benchmark_retrievers(args)
    for label, path in parse_trace_specs(args.trace):
        rows.extend(summarize_trace(label, path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["system", "query_id", "instrument", "latency_seconds", "source", "top_k"],
        )
        writer.writeheader()
        writer.writerows(rows)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
