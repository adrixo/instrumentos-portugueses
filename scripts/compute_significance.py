#!/usr/bin/env python3
"""Significance testing for the slide deck (ADR §6.4 / §18.3).

Reuses `instrument_ir.evaluation.statistical_tests` (bootstrap CI + paired
permutation test + Holm-Bonferroni). Reads the per-query metrics already stored
in each run's `metrics/*.json` — no model re-execution needed.

Outputs (consumed by figures and slides):
  results/esalab-big/2026-06-30_gpu_full/outputs/reports/tables/significance.md
  results/esalab-big/2026-06-30_gpu_full/outputs/reports/tables/significance.json
"""

from __future__ import annotations

import json
from pathlib import Path

from instrument_ir.evaluation.statistical_tests import (
    bootstrap_ci,
    compare_systems,
    holm_bonferroni,
)

ESALAB = Path("results/esalab-big/2026-06-30_gpu_full/outputs/metrics")
QWEN = Path("results/mida-qwen36-27b/2026-06-30_zero_shot_top100/outputs/metrics")
OUT_DIR = Path("results/esalab-big/2026-06-30_gpu_full/outputs/reports/tables")

SEED = 42
METRICS = ["recall@100", "ndcg@10", "ndcg@100", "map", "mrr"]

# system id -> (metrics dir, human label)
SYSTEMS = {
    "B1_openclip-vitl14_test": (ESALAB, "OpenCLIP L/14 (dense)"),
    "B1_jinaclip_test": (ESALAB, "JinaCLIP (dense)"),
    "B3_colqwen_test": (ESALAB, "ColQwen (late-interaction)"),
    "B4_test": (ESALAB, "VLM-rerank (Qwen2.5-VL-3B)"),
    "B5_full_test": (ESALAB, "Agentic"),
    "B4_qwen36_zero_shot_top100_test": (QWEN, "VLM-rerank (Qwen3.6-27B, top-100)"),
}

# (better, baseline, note) — better vs baseline, so delta = better - baseline
PAIRS = [
    ("B1_jinaclip_test", "B1_openclip-vitl14_test", "mejor dense vs dense base"),
    ("B3_colqwen_test", "B1_openclip-vitl14_test", "late-interaction vs dense"),
    ("B4_test", "B1_openclip-vitl14_test", "VLM-rerank (3B) vs su dense base"),
    ("B5_full_test", "B4_test", "agentic vs VLM-rerank pointwise"),
    ("B4_qwen36_zero_shot_top100_test", "B1_openclip-vitl14_test", "VLM grande (27B) vs dense base"),
    (
        "B4_qwen36_zero_shot_top100_test",
        "B4_test",
        "VLM grande (27B, top-100) vs pequeño (3B, top-200) — CONFOUND profundidad",
    ),
]


def per_query_metric(system: str, metric: str) -> dict[str, float]:
    root, _ = SYSTEMS[system]
    data = json.loads((root / f"{system}.json").read_text(encoding="utf-8"))
    return {q: v[metric] for q, v in data.get("per_query", {}).items() if metric in v}


def main() -> None:
    result: dict = {"seed": SEED, "metrics": METRICS, "systems": {}, "comparisons": {}}

    # Per-system 95% CI (for figure error bars).
    for system, (_, label) in SYSTEMS.items():
        result["systems"][system] = {"label": label, "ci": {}}
        for metric in METRICS:
            vals = list(per_query_metric(system, metric).values())
            result["systems"][system]["ci"][metric] = bootstrap_ci(vals, seed=SEED)

    # Pairwise comparisons, Holm-corrected within each metric.
    md_blocks: list[str] = [
        "# Significancia estadística (n=66 consultas, bootstrap 10k, permutación pareada 10k)\n",
        "IC 95% del delta por remuestreo de consultas. p ajustado por Holm-Bonferroni dentro de cada métrica.\n",
    ]
    for metric in METRICS:
        rows, pvals = [], {}
        for better, base, note in PAIRS:
            cmp = compare_systems(
                per_query_metric(better, metric),
                per_query_metric(base, metric),
                seed=SEED,
            )
            name = f"{SYSTEMS[better][1]} vs {SYSTEMS[base][1]}"
            rows.append((name, note, cmp))
            pvals[name] = cmp["p_value"]
            result["comparisons"].setdefault(metric, []).append(
                {
                    "better": better,
                    "baseline": base,
                    "note": note,
                    "delta": cmp["delta_ci"]["mean"],
                    "ci_lo": cmp["delta_ci"]["lo"],
                    "ci_hi": cmp["delta_ci"]["hi"],
                    "p_value": cmp["p_value"],
                }
            )
        adj = holm_bonferroni(pvals)
        # attach adjusted p / significance back into json
        for entry, (name, _, _) in zip(result["comparisons"][metric], rows):
            a = adj.get(name, {})
            entry["p_holm"] = a.get("p_adj", entry["p_value"])
            entry["significant"] = bool(a.get("significant", False))

        md_blocks.append(f"\n## {metric}\n")
        md_blocks.append("| comparación | delta | IC 95% | p (Holm) | signif. |")
        md_blocks.append("|---|---|---|---|---|")
        for name, note, cmp in rows:
            d = cmp["delta_ci"]
            a = adj.get(name, {})
            sig = "**sí**" if a.get("significant") else "no"
            md_blocks.append(
                f"| {name} | {d['mean']:+.4f} | [{d['lo']:+.3f}, {d['hi']:+.3f}] | "
                f"{a.get('p_adj', cmp['p_value']):.3f} | {sig} |"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "significance.md").write_text("\n".join(md_blocks) + "\n", encoding="utf-8")
    (OUT_DIR / "significance.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {OUT_DIR/'significance.md'}")
    print(f"Wrote {OUT_DIR/'significance.json'}")

    # Console summary: how many comparisons are significant?
    n_sig = sum(
        1 for m in result["comparisons"].values() for e in m if e["significant"]
    )
    n_tot = sum(len(m) for m in result["comparisons"].values())
    print(f"Significant (Holm, α=0.05): {n_sig}/{n_tot} comparaciones")


if __name__ == "__main__":
    main()
