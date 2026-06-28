import json
from pathlib import Path

from PIL import Image

from instrument_ir.agent.crop_generation import generate_crops
from instrument_ir.agent.graph import AgenticReranker
from instrument_ir.data.queries import Query
from instrument_ir.reranking.base import Candidate
from instrument_ir.reranking.vlm_backend import MockVLMBackend


class FakeProvider:
    def load(self, image_id):
        return Image.new("RGB", (64, 64))


def test_generate_crops_pure():
    img = Image.new("RGB", (100, 100))
    crops = generate_crops(img, ["center", "grid2x2"], max_crops=5)
    assert [cid for cid, _ in crops] == [f"crop_{i:02d}" for i in range(5)]
    # cada crop es una imagen no vacía
    assert all(c.size[0] > 0 and c.size[1] > 0 for _, c in crops)


def test_agent_uses_crops_when_uncertain_and_fuses_max():
    # full=uncertain(0.5) -> dispara crops; caption; crop=present(0.9) -> fusión max = 0.9
    scripted = [
        '{"decision":"uncertain","confidence":0.5,"visual_evidence":["maybe"],"negative_evidence":[],"score":0.25}',
        "an adufe square frame drum held by a performer",
        '{"decision":"present","confidence":0.9,"visual_evidence":["square frame drum"],"negative_evidence":[],"score":0.9}',
    ]
    agent = AgenticReranker(
        MockVLMBackend(scripted), FakeProvider(),
        crop_strategies=["center"], max_crops=1, fusion="max", high_confidence_threshold=0.8,
    )
    q = Query("q_adufe_en", "adufe", "adufe", "en")
    cands = [Candidate("img_valid_000000", 1, 0.3)]
    reranked, traces = agent.rerank(q, {"canonical_name": "adufe"}, cands, "B5_test")

    assert reranked[0].final_score == 0.9
    assert reranked[0].decision == "present"
    tools = [s["tool"] for s in traces[0]["steps"]]
    assert tools == ["ask_vlm_full_image", "caption_image", "generate_crops", "ask_vlm_crop", "score_evidence"]
    assert traces[0]["model_versions"]["agent_framework"] == "custom_graph"


def test_agent_full_image_only_skips_crops():
    scripted = ['{"decision":"present","confidence":0.95,"visual_evidence":["x"],"negative_evidence":[],"score":0.95}']
    agent = AgenticReranker(MockVLMBackend(scripted), FakeProvider(), full_image_only=True)
    q = Query("q_adufe_en", "adufe", "adufe", "en")
    _, traces = agent.rerank(q, {"canonical_name": "adufe"}, [Candidate("img_valid_000000", 1, 0.3)], "B5_fio")
    assert [s["tool"] for s in traces[0]["steps"]] == ["ask_vlm_full_image"]


def test_b5_trace_validates_against_schema():
    import jsonschema

    schema = json.loads(Path("schemas/agent_trace.schema.json").read_text())
    scripted = [
        '{"decision":"uncertain","confidence":0.5,"visual_evidence":[],"negative_evidence":[],"score":0.25}',
        "a cavaquinho",
        '{"decision":"present","confidence":0.8,"visual_evidence":["small guitar"],"negative_evidence":[],"score":0.8}',
    ]
    agent = AgenticReranker(
        MockVLMBackend(scripted), FakeProvider(), crop_strategies=["center"], max_crops=1,
    )
    q = Query("q_cavaquinho_en", "cavaquinho", "cavaquinho", "en")
    _, traces = agent.rerank(q, {"canonical_name": "cavaquinho"}, [Candidate("img_valid_000007", 5, 0.2)], "B5_schema")
    jsonschema.validate(traces[0], schema)  # lanza si no valida
