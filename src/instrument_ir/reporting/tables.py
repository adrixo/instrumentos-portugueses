"""Tablas de resultados (ADR §13, §18). Markdown + LaTeX, sin dependencias pesadas.

Lee los JSON de outputs/metrics/ y produce tablas macro y por instrumento.
"""

from __future__ import annotations

import json
from pathlib import Path

# Métricas mostradas en la tabla macro (ADR §18.1), si están disponibles.
MACRO_COLUMNS = ["recall@20", "recall@50", "recall@100", "ndcg@10", "ndcg@100", "map", "mrr"]


def load_all_metrics(metrics_dir: Path) -> dict[str, dict]:
    """Devuelve {system_name: metrics_json} para cada *.json del directorio."""
    out: dict[str, dict] = {}
    for path in sorted(Path(metrics_dir).glob("*.json")):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _system_metrics(data: dict) -> dict:
    """Prefiere macro por instrumento (más honesto con el desbalance); si no, micro."""
    return data.get("macro_metrics", data.get("metrics", {}))


def macro_table_md(all_metrics: dict[str, dict], columns: list[str] = MACRO_COLUMNS) -> str:
    cols = [c for c in columns if any(c in _system_metrics(d) for d in all_metrics.values())]
    header = "| system | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [header, sep]
    for system, data in all_metrics.items():
        m = _system_metrics(data)
        cells = [f"{m.get(c, float('nan')):.4f}" if c in m else "—" for c in cols]
        rows.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def macro_table_latex(all_metrics: dict[str, dict], columns: list[str] = MACRO_COLUMNS) -> str:
    cols = [c for c in columns if any(c in _system_metrics(d) for d in all_metrics.values())]
    lines = [
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        "system & " + " & ".join(c.replace("@", "@") for c in cols) + " \\\\",
        "\\midrule",
    ]
    for system, data in all_metrics.items():
        m = _system_metrics(data)
        cells = [f"{m.get(c, float('nan')):.4f}" if c in m else "--" for c in cols]
        lines.append(system.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def per_instrument_table_md(all_metrics: dict[str, dict], metric: str = "recall@100") -> str:
    """Tabla instrumento × sistema para una métrica (ADR §18.2)."""
    systems = [s for s, d in all_metrics.items() if "per_instrument" in d]
    if not systems:
        return "_(sin datos per-instrument)_"
    instruments = sorted({
        ins for s in systems for ins in all_metrics[s]["per_instrument"]
    })
    header = "| instrument | " + " | ".join(systems) + " | best |"
    sep = "|" + "---|" * (len(systems) + 2)
    rows = [header, sep]
    for ins in instruments:
        vals = {}
        for s in systems:
            pi = all_metrics[s]["per_instrument"].get(ins, {})
            vals[s] = pi.get(metric)
        cells = [f"{vals[s]:.3f}" if vals[s] is not None else "—" for s in systems]
        best = max((s for s in systems if vals[s] is not None), key=lambda s: vals[s], default="—")
        rows.append(f"| {ins} | " + " | ".join(cells) + f" | {best} |")
    return "\n".join(rows)
