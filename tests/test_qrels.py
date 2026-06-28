from instrument_ir.data.anonymize import build_mapping
from instrument_ir.data.coco_parser import parse_split
from instrument_ir.data.qrels import build_qrels, load_qrels_trec, write_qrels_trec
from instrument_ir.data.queries import Query


def _queries():
    return [
        Query("q_adufe_en", "adufe", "adufe", "en"),
        Query("q_cavaquinho_en", "cavaquinho", "cavaquinho", "en"),
    ]


def test_qrels_multilabel(coco_root):
    coco = parse_split(coco_root, "valid")
    mapping = build_mapping(coco)
    rows = build_qrels(mapping, _queries())

    by_query: dict[str, set[str]] = {}
    for qid, iid, rel in rows:
        assert rel == 1  # solo se escriben relevantes
        by_query.setdefault(qid, set()).add(iid)

    # img_valid_000000 (multi-etiqueta) es relevante para ambas queries.
    assert "img_valid_000000" in by_query["q_adufe_en"]
    assert "img_valid_000000" in by_query["q_cavaquinho_en"]
    # adufe: img0 e img2 ; cavaquinho: img0 e img1
    assert by_query["q_adufe_en"] == {"img_valid_000000", "img_valid_000002"}
    assert by_query["q_cavaquinho_en"] == {"img_valid_000000", "img_valid_000001"}


def test_qrels_trec_roundtrip(coco_root, tmp_path):
    coco = parse_split(coco_root, "valid")
    mapping = build_mapping(coco)
    rows = build_qrels(mapping, _queries())
    path = tmp_path / "valid.qrels"
    write_qrels_trec(rows, path)

    loaded = load_qrels_trec(path)
    assert loaded["q_adufe_en"]["img_valid_000000"] == 1
    # formato TREC: 4 columnas por línea
    for line in path.read_text().splitlines():
        assert len(line.split()) == 4
