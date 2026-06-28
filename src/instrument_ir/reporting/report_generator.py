"""Generador de informe final (ADR §13). Markdown + HTML simple + tablas + figuras."""

from __future__ import annotations

from pathlib import Path

from .plots import recall_at_k_plot
from .tables import load_all_metrics, macro_table_latex, macro_table_md, per_instrument_table_md

SECTIONS_INTRO = """# Instrument Retrieval Lab — Informe de resultados

## 1. Objetivo
Recuperación visual de instrumentos tradicionales portugueses: dada una query textual, ordenar las
imágenes que contienen el instrumento, sin usar nombres de archivo, rutas ni etiquetas en inferencia.

## 2. Dataset
22 instrumentos (categoría paraguas excluida), splits train/valid/test, ground truth COCO
multi-etiqueta. DOI Mendeley 10.17632/pk7txkgt4v.2.

## 3. Prevención de fuga
image_id anónimos, mapping privado, prompts/traces sin filename/vimeo/labels. Tests automáticos.

## 4. Métodos
B1 dense global · B3 late-interaction (ColQwen) · B4 dense+VLM reranker · B5 dense+agente determinista.
"""


def generate_report(
    metrics_dir: Path = Path("outputs/metrics"),
    out_dir: Path = Path("outputs/reports"),
    figures: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    all_metrics = load_all_metrics(metrics_dir)

    macro_md = macro_table_md(all_metrics)
    (out_dir / "tables" / "results_macro.md").write_text(macro_md, encoding="utf-8")
    (out_dir / "tables" / "results_macro.tex").write_text(
        macro_table_latex(all_metrics), encoding="utf-8"
    )
    per_instr_md = per_instrument_table_md(all_metrics)
    (out_dir / "tables" / "results_per_class.md").write_text(per_instr_md, encoding="utf-8")

    fig_line = ""
    if figures:
        fig_path = recall_at_k_plot(all_metrics, out_dir / "figures" / "recall_at_k.png")
        if fig_path:
            fig_line = f"\n![Recall@K](figures/{fig_path.name})\n"

    report = (
        f"{SECTIONS_INTRO}\n"
        f"## 5. Resultados (macro por instrumento)\n\n{macro_md}\n{fig_line}\n"
        f"## 6. Resultados por instrumento (Recall@100)\n\n{per_instr_md}\n\n"
        f"## 7. Sistemas evaluados\n\n{len(all_metrics)} runs en `{metrics_dir}`.\n\n"
        f"## 8. Limitaciones\n"
        f"- Modelos fundacionales pueden conocer instrumentos comunes.\n"
        f"- ~22 clases → potencia estadística limitada; CIs anchos.\n"
        f"- B4/B5 dependen del recall inicial del dense (oracle_recall@N).\n"
    )
    md_path = out_dir / "final_report.md"
    md_path.write_text(report, encoding="utf-8")

    # HTML simple (sin dependencias): envoltorio mínimo del markdown.
    html = f"<!doctype html><meta charset='utf-8'><title>Instrument Retrieval Lab</title>\n<pre>\n{report}\n</pre>"
    (out_dir / "final_report.html").write_text(html, encoding="utf-8")
    return md_path
