"""Figuras de resultados (ADR §13). matplotlib es opcional (extra [extras]); si no está, se omite."""

from __future__ import annotations

from pathlib import Path

RECALL_KS = [10, 20, 50, 100]


def _system_metrics(data: dict) -> dict:
    return data.get("macro_metrics", data.get("metrics", {}))


def recall_at_k_plot(all_metrics: dict[str, dict], out_path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for system, data in all_metrics.items():
        m = _system_metrics(data)
        ys = [m.get(f"recall@{k}") for k in RECALL_KS]
        if all(y is not None for y in ys):
            ax.plot(RECALL_KS, ys, marker="o", label=system)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K (macro)")
    ax.set_title("Recall@K por sistema")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
