import json

from instrument_ir.reporting.report_generator import generate_report
from instrument_ir.reporting.slides_generator import generate_slides
from instrument_ir.reporting.tables import macro_table_md, per_instrument_table_md


def _metrics_dir(tmp_path):
    d = tmp_path / "metrics"
    d.mkdir()
    (d / "B1_openclip_valid.json").write_text(json.dumps({
        "macro_metrics": {"recall@100": 0.18, "ndcg@10": 0.19, "map": 0.07, "mrr": 0.29},
        "per_instrument": {"adufe": {"recall@100": 0.19}, "concertina": {"recall@100": 0.77}},
    }))
    (d / "B4_vlm_valid.json").write_text(json.dumps({
        "macro_metrics": {"recall@100": 0.25, "ndcg@10": 0.30, "map": 0.12, "mrr": 0.40},
        "per_instrument": {"adufe": {"recall@100": 0.30}, "concertina": {"recall@100": 0.80}},
    }))
    return d


def test_macro_table_md(tmp_path):
    from instrument_ir.reporting.tables import load_all_metrics

    m = load_all_metrics(_metrics_dir(tmp_path))
    table = macro_table_md(m)
    assert "| system |" in table
    assert "B1_openclip_valid" in table and "B4_vlm_valid" in table
    assert "recall@100" in table


def test_per_instrument_best(tmp_path):
    from instrument_ir.reporting.tables import load_all_metrics

    m = load_all_metrics(_metrics_dir(tmp_path))
    table = per_instrument_table_md(m, metric="recall@100")
    # B4 mejora a B1 en ambos instrumentos -> best = B4
    assert "adufe" in table and "concertina" in table
    assert table.count("B4_vlm_valid") >= 2  # best en las dos filas


def test_generate_report_and_slides(tmp_path):
    md_dir = _metrics_dir(tmp_path)
    out = tmp_path / "reports"
    report = generate_report(md_dir, out, figures=False)
    assert report.exists()
    assert (out / "tables" / "results_macro.md").exists()
    assert (out / "tables" / "results_macro.tex").exists()
    assert (out / "final_report.html").exists()

    deck = generate_slides(md_dir, tmp_path / "slides.md")
    text = deck.read_text()
    assert "marp: true" in text and "Resultados principales" in text
