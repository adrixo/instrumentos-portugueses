#!/usr/bin/env python3
"""Build a Qwen3.6 top-100 rerank by reusing a completed top-50 trace.

The normal B4 runner is intentionally simple and starts every run from scratch.
For the Qwen3.6 comparison we already have top-50 traces, so this script reuses
those decisions and only calls the VLM for candidates not present in the previous
trace. It also sets an explicit OpenAI-compatible request timeout so one stalled
llama.cpp request cannot block the whole run indefinitely.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from instrument_ir.data.prepare_dataset import resolve_mapping
from instrument_ir.data.queries import load_instruments, load_queries
from instrument_ir.retrieval.base import ScoredDoc
from instrument_ir.reranking.base import RerankedDoc, load_candidates_from_run
from instrument_ir.reranking.prompts import build_rerank_prompt
from instrument_ir.reranking.vlm_backend import OpenAICompatVLMBackend
from instrument_ir.reranking.vlm_pointwise import VLMPointwiseReranker
from instrument_ir.utils.io import ImageProvider
from instrument_ir.utils.trec import write_run_trec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.6 top-100 resume reranker")
    parser.add_argument("--dense-run", type=Path, default=Path("outputs/runs/DENSE_qwen36_top100_test.trec"))
    parser.add_argument("--previous-trace", type=Path, default=Path("outputs/rerank_traces/B4_qwen36_zero_shot_test.jsonl"))
    parser.add_argument("--run-name", default="B4_qwen36_zero_shot_top100_test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--final-top-k", type=int, default=100)
    parser.add_argument("--base-url", default="http://100.127.120.42:8080/v1")
    parser.add_argument("--vlm-model", default="qwen36-27b")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/portuguese_instruments"))
    parser.add_argument("--queries", type=Path, default=Path("configs/queries.yaml"))
    parser.add_argument("--instruments", type=Path, default=Path("configs/instruments.yaml"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    return parser.parse_args()


def load_previous(path: Path) -> dict[tuple[str, str], dict]:
    reused: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return reused
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        query_id = item.get("query_id")
        image_id = item.get("image_id")
        if query_id and image_id:
            reused[(query_id, image_id)] = item
    return reused


def doc_from_trace(trace: dict, dense_rank: int, dense_score: float) -> RerankedDoc:
    decision = str(trace.get("vlm_decision", "uncertain"))
    confidence = float(trace.get("vlm_confidence", 0.0) or 0.0)
    final_score = float(trace.get("final_score", trace.get("vlm_score", 0.0)) or 0.0)
    return RerankedDoc(
        image_id=str(trace["image_id"]),
        final_score=final_score,
        final_rank=-1,
        dense_rank=dense_rank,
        dense_score=dense_score,
        decision=decision,
        confidence=confidence,
    )


def reused_trace(source: dict, run_id: str, dense_rank: int, dense_score: float) -> dict:
    trace = dict(source)
    trace["run_id"] = run_id
    trace["dense_rank"] = dense_rank
    trace["dense_score"] = dense_score
    trace["reused_from_run"] = source.get("run_id")
    return trace


def timeout_trace(run_id: str, query_id: str, instrument_id: str, cand, exc: Exception) -> tuple[RerankedDoc, dict]:
    doc = RerankedDoc(
        image_id=cand.image_id,
        final_score=0.0,
        final_rank=-1,
        dense_rank=cand.dense_rank,
        dense_score=cand.dense_score,
        decision="uncertain",
        confidence=0.0,
    )
    trace = {
        "run_id": run_id,
        "query_id": query_id,
        "instrument": instrument_id,
        "image_id": cand.image_id,
        "dense_rank": cand.dense_rank,
        "dense_score": cand.dense_score,
        "vlm_decision": "uncertain",
        "vlm_confidence": 0.0,
        "vlm_score": 0.0,
        "final_score": 0.0,
        "visual_evidence": [],
        "negative_evidence": [f"{type(exc).__name__}: {exc}"],
        "model": "qwen36-27b",
        "temperature": 0.0,
        "seed": 42,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return doc, trace


def main() -> None:
    args = parse_args()
    mapping = resolve_mapping(args.processed, args.split)
    provider = ImageProvider(mapping, args.raw)
    queries = {q.query_id: q for q in load_queries(args.queries)}
    instruments = load_instruments(args.instruments)
    candidates_by_query = load_candidates_from_run(args.dense_run, args.top_n)
    previous = load_previous(args.previous_trace)

    backend = OpenAICompatVLMBackend(args.vlm_model, args.base_url, api_key="local")
    if args.timeout > 0:
        from openai import OpenAI

        backend.client = OpenAI(base_url=args.base_url, api_key="local", timeout=args.timeout)
    reranker = VLMPointwiseReranker(backend, provider, seed=42)

    traces_path = args.outputs / "rerank_traces" / f"{args.run_name}.jsonl"
    cand_path = args.outputs / "candidates" / f"{args.run_name}.parquet"
    run_path = args.outputs / "runs" / f"{args.run_name}.trec"
    for path in (traces_path, cand_path, run_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    total = sum(len(cands) for cands in candidates_by_query.values())
    processed = 0
    reused_count = 0
    new_count = 0
    rankings: dict[str, list[ScoredDoc]] = {}
    cand_rows: list[dict] = []

    with traces_path.open("w", encoding="utf-8") as tf:
        for query_id, cands in candidates_by_query.items():
            query = queries[query_id]
            instrument = instruments.get(query.instrument_id, {"canonical_name": query.instrument_id})
            prompt = build_rerank_prompt(instrument)
            docs: list[RerankedDoc] = []
            traces: list[dict] = []

            for cand in cands:
                cand_rows.append(
                    {
                        "query_id": query_id,
                        "image_id": cand.image_id,
                        "dense_rank": cand.dense_rank,
                        "dense_score": cand.dense_score,
                        "run_id": args.run_name,
                    }
                )
                prev = previous.get((query_id, cand.image_id))
                if prev is not None:
                    doc = doc_from_trace(prev, cand.dense_rank, cand.dense_score)
                    trace = reused_trace(prev, args.run_name, cand.dense_rank, cand.dense_score)
                    reused_count += 1
                else:
                    try:
                        doc, trace = reranker._process_candidate(prompt, query, cand, args.run_name)
                    except Exception as exc:  # keep the run moving; diagnostics go into the trace.
                        doc, trace = timeout_trace(args.run_name, query_id, query.instrument_id, cand, exc)
                    new_count += 1
                docs.append(doc)
                traces.append(trace)

            order = sorted(
                range(len(docs)),
                key=lambda i: (docs[i].final_score, docs[i].dense_score),
                reverse=True,
            )
            reranked: list[RerankedDoc] = []
            for rank, index in enumerate(order, start=1):
                doc = docs[index]
                reranked.append(
                    RerankedDoc(
                        image_id=doc.image_id,
                        final_score=doc.final_score,
                        final_rank=rank,
                        dense_rank=doc.dense_rank,
                        dense_score=doc.dense_score,
                        decision=doc.decision,
                        confidence=doc.confidence,
                    )
                )
                traces[index]["final_rank"] = rank

            for trace in traces:
                tf.write(json.dumps(trace, ensure_ascii=False) + "\n")
            tf.flush()
            processed += len(traces)
            print(
                f"rerank progress: {processed}/{total} candidates "
                f"({query_id}; reused={reused_count}, new={new_count})",
                flush=True,
            )
            rankings[query_id] = [
                ScoredDoc(doc.image_id, doc.final_score) for doc in reranked[: args.final_top_k]
            ]

    pd.DataFrame(cand_rows).to_parquet(cand_path, index=False)
    write_run_trec(rankings, args.run_name, run_path)
    print(f"B4 runfile: {run_path}")
    print(f"  traces: {traces_path}")
    print(f"  candidates: {cand_path}")
    print(f"  reused traces: {reused_count}")
    print(f"  new VLM calls: {new_count}")


if __name__ == "__main__":
    main()
