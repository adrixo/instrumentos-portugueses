from PIL import Image

from instrument_ir.data.queries import Query
from instrument_ir.reranking.base import Candidate
from instrument_ir.reranking.schemas import parse_vlm_response
from instrument_ir.reranking.scoring import normalize_score
from instrument_ir.reranking.vlm_backend import MockVLMBackend
from instrument_ir.reranking.vlm_pointwise import VLMPointwiseReranker


class FakeProvider:
    def load(self, image_id):
        return Image.new("RGB", (8, 8))


def test_normalize_score_rule():
    assert normalize_score("present", 0.9) == 0.9
    assert normalize_score("uncertain", 0.6) == 0.3
    assert normalize_score("absent", 0.9) == 0.0
    assert normalize_score("present", 5.0) == 1.0  # clamp


def test_parse_vlm_response_robustness():
    good = '{"decision":"present","confidence":0.8,"visual_evidence":["x"],"negative_evidence":[],"score":0.8}'
    assert parse_vlm_response(good)["decision"] == "present"
    wrapped = 'Sure!\n{"decision":"absent","confidence":0.1,"visual_evidence":[],"negative_evidence":["none"],"score":0}'
    assert parse_vlm_response(wrapped)["decision"] == "absent"
    assert parse_vlm_response("no json here")["decision"] == "uncertain"  # degradación segura


def test_vlm_pointwise_reorders_and_traces():
    cands = [
        Candidate("img_valid_000000", dense_rank=1, dense_score=0.9),  # absent
        Candidate("img_valid_000001", dense_rank=2, dense_score=0.5),  # present 0.9
        Candidate("img_valid_000002", dense_rank=3, dense_score=0.1),  # uncertain 0.6 -> 0.3
    ]
    scripted = [
        '{"decision":"absent","confidence":0.2,"visual_evidence":[],"negative_evidence":["no"],"score":0}',
        '{"decision":"present","confidence":0.9,"visual_evidence":["frame drum"],"negative_evidence":[],"score":0.9}',
        '{"decision":"uncertain","confidence":0.6,"visual_evidence":["maybe"],"negative_evidence":[],"score":0.3}',
    ]
    reranker = VLMPointwiseReranker(MockVLMBackend(scripted), FakeProvider(), seed=42)
    query = Query("q_adufe_en", "adufe", "adufe", "en")
    reranked, traces = reranker.rerank(query, {"canonical_name": "adufe"}, cands, "B4_test")

    order = [d.image_id for d in reranked]
    assert order == ["img_valid_000001", "img_valid_000002", "img_valid_000000"]
    assert reranked[0].final_rank == 1 and reranked[0].final_score == 0.9
    assert reranked[1].final_score == 0.3

    # trazas: una por candidato, con final_rank asignado y sin fuga de filename
    assert len(traces) == 3
    assert all("final_rank" in t for t in traces)
    assert all("image_id" in t and t["image_id"].startswith("img_valid_") for t in traces)
    assert all("file_name" not in t for t in traces)
