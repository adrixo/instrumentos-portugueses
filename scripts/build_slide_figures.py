#!/usr/bin/env python3
"""Build reproducible figures consumed by the Slidev deck."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS = Path("results/esalab-big/2026-06-30_gpu_full/outputs")
QWEN_RESULTS = Path("results/mida-qwen36-27b/2026-06-30_zero_shot_top50/outputs")
ASSETS = Path("slides/slidev/assets")

SYSTEMS = [
    (RESULTS, "B1_openclip-vitb32_test", "OpenCLIP B/32"),
    (RESULTS, "B1_openclip-vitl14_test", "OpenCLIP L/14"),
    (RESULTS, "B1_jinaclip_test", "JinaCLIP"),
    (RESULTS, "B3_colqwen_test", "ColQwen"),
    (RESULTS, "B4_test", "VLM rerank"),
    (RESULTS, "B5_full_test", "Agéntico"),
    (QWEN_RESULTS, "B4_qwen36_zero_shot_test", "Qwen3.6 VLM top-50"),
]

PALETTE = {
    "OpenCLIP B/32": "#7c8a96",
    "OpenCLIP L/14": "#4f6f8f",
    "JinaCLIP": "#155e75",
    "ColQwen": "#7c3aed",
    "VLM rerank": "#0f766e",
    "Agéntico": "#b45309",
    "Qwen3.6 VLM top-50": "#be123c",
}

LATENCY_LABELS = {
    "openclip-vitl14": "OpenCLIP L/14",
    "jinaclip": "JinaCLIP",
    "colqwen": "ColQwen",
    "B4_VLM": "B4 VLM",
    "B5_agentic": "B5 agéntico",
    "B4_qwen36_zero_shot": "Qwen3.6 VLM",
}


def load_metrics() -> pd.DataFrame:
    rows = []
    for root, system, label in SYSTEMS:
        path = root / "metrics" / f"{system}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"system": label}
        row.update(data["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def style_axes(ax):
    ax.grid(axis="y", color="#d6e2df", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#263735", labelsize=9)


def save_recall_at_k(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=180)
    ks = [10, 20, 50, 100]
    for _, row in df.iterrows():
        label = row["system"]
        ax.plot(
            ks,
            [row[f"recall@{k}"] for k in ks],
            marker="o",
            linewidth=2.2,
            label=label,
            color=PALETTE[label],
        )
    ax.set_title("Recall@K por sistema", fontsize=16, weight="bold", color="#10201f")
    ax.set_xlabel("K", color="#50615f")
    ax.set_ylabel("Recall", color="#50615f")
    ax.set_ylim(0, 0.22)
    ax.set_xticks(ks)
    style_axes(ax)
    ax.legend(ncol=2, frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "metrics_recall_at_k.png", bbox_inches="tight")
    plt.close(fig)


def save_quality_bars(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.5), dpi=180, sharey=False)
    metrics = [("ndcg@10", "nDCG@10"), ("map", "mAP"), ("mrr", "MRR")]
    for ax, (metric, title) in zip(axes, metrics):
        values = df[metric].tolist()
        labels = df["system"].tolist()
        colors = [PALETTE[label] for label in labels]
        ax.barh(labels, values, color=colors, height=0.58)
        ax.set_title(title, fontsize=13, weight="bold", color="#10201f")
        ax.set_xlim(0, max(values) * 1.18)
        ax.invert_yaxis()
        style_axes(ax)
        for i, value in enumerate(values):
            ax.text(value + max(values) * 0.02, i, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle("Calidad del ranking en el split de test", fontsize=16, weight="bold", color="#10201f")
    fig.tight_layout()
    fig.savefig(ASSETS / "metrics_quality_bars.png", bbox_inches="tight")
    plt.close(fig)


def save_latency_boxplot() -> None:
    path = RESULTS / "metrics" / "query_latency.csv"
    if not path.exists():
        path = RESULTS / "metrics" / "query_latency_traces.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    qwen_trace = QWEN_RESULTS / "rerank_traces" / "B4_qwen36_zero_shot_test.jsonl"
    rows.extend(trace_latency_rows("B4_qwen36_zero_shot", qwen_trace, top_k="50"))
    by_system: dict[str, list[float]] = {}
    for row in rows:
        by_system.setdefault(LATENCY_LABELS.get(row["system"], row["system"]), []).append(
            float(row["latency_seconds"])
        )
    if not by_system:
        return
    order = ["OpenCLIP L/14", "JinaCLIP", "ColQwen", "B4 VLM", "B5 agéntico", "Qwen3.6 VLM"]
    labels = [label for label in order if label in by_system]
    labels.extend(label for label in by_system if label not in labels)
    values = [by_system[label] for label in labels]

    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=180)
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    colors = {
        "OpenCLIP L/14": "#4f6f8f",
        "JinaCLIP": "#155e75",
        "ColQwen": "#7c3aed",
        "B4 VLM": "#0f766e",
        "B5 agéntico": "#b45309",
        "Qwen3.6 VLM": "#be123c",
    }
    for patch, label in zip(box["boxes"], labels):
        color = colors.get(label, "#6b7280")
        patch.set_facecolor(color)
        patch.set_alpha(0.20)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(2)
    ax.set_title("Latencia observada por consulta", fontsize=16, weight="bold", color="#10201f")
    ax.set_ylabel("Segundos por consulta", color="#50615f")
    ax.set_yscale("log")
    style_axes(ax)
    ax.text(
        0.02,
        0.97,
        "Escala log. Dense/ColQwen: benchmark con caché; B4/B5: trazas top-200; Qwen3.6: traza top-50",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#50615f",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "latency_boxplot.png", bbox_inches="tight")
    plt.close(fig)


def trace_latency_rows(label: str, path: Path, top_k: str) -> list[dict]:
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
                "top_k": top_k,
            }
        )
    return rows


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    save_recall_at_k(df)
    save_quality_bars(df)
    save_latency_boxplot()


if __name__ == "__main__":
    main()
