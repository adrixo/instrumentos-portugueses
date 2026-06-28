from instrument_ir.retrieval.base import ScoredDoc
from instrument_ir.utils.trec import load_run_trec, write_run_trec


def test_runfile_trec_format(tmp_path):
    rankings = {
        "q_adufe_en": [ScoredDoc("img_valid_000000", 0.9), ScoredDoc("img_valid_000002", 0.4)],
        "q_cavaquinho_en": [ScoredDoc("img_valid_000001", 0.7)],
    }
    path = tmp_path / "run.trec"
    write_run_trec(rankings, "B1_test", path)

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        cols = line.split()
        # query_id Q0 image_id rank score run_name
        assert len(cols) == 6
        assert cols[1] == "Q0"
        assert cols[5] == "B1_test"
        int(cols[3])     # rank entero
        float(cols[4])   # score float

    # rank empieza en 1 y es creciente por query
    assert lines[0].split()[3] == "1"
    assert lines[1].split()[3] == "2"


def test_runfile_roundtrip(tmp_path):
    rankings = {"q1": [ScoredDoc("img_valid_000000", 0.5)]}
    path = tmp_path / "run.trec"
    write_run_trec(rankings, "r", path)
    run = load_run_trec(path)
    assert run["q1"]["img_valid_000000"] == 0.5
