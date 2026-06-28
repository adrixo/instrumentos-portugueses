import pandas as pd

from instrument_ir.serving.search import SearchService


def _setup(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "B1_dummy_valid.trec").write_text(
        "q_adufe_en Q0 img_valid_000000 1 0.9 B1\n"
        "q_adufe_en Q0 img_valid_000001 2 0.5 B1\n"
    )
    qd = tmp_path / "qrels"
    qd.mkdir()
    (qd / "valid.qrels").write_text("q_adufe_en 0 img_valid_000000 1\n")
    mapping = pd.DataFrame([
        {"image_id": "img_valid_000000", "vimeo_id": "100200300", "split": "valid",
         "file_name": "100200300_Adufe_25_jpg.rf.x.jpg"},
        {"image_id": "img_valid_000001", "vimeo_id": "999888777", "split": "valid",
         "file_name": "999888777_Cavaquinho_25_jpg.rf.y.jpg"},
    ])
    mp = tmp_path / "mapping.parquet"
    mapping.to_parquet(mp)
    return SearchService(runs_dir=runs, mapping_path=mp, raw_root=tmp_path / "raw", qrels_dir=qd)


def test_search_returns_vimeo_links_and_relevance(tmp_path):
    svc = _setup(tmp_path)
    assert svc.list_runs() == ["B1_dummy_valid"]
    res = svc.search("B1_dummy_valid", "q_adufe_en", top_k=10)
    assert [r["rank"] for r in res] == [1, 2]
    assert res[0]["image_id"] == "img_valid_000000"
    assert res[0]["vimeo_url"] == "https://vimeo.com/100200300"
    assert res[0]["relevant"] is True       # está en qrels
    assert res[1]["relevant"] is False
