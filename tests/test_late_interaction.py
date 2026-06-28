import numpy as np

from instrument_ir.data.queries import Query
from instrument_ir.retrieval.late_interaction import (
    LateInteractionRetriever,
    MockMultiVectorEmbedder,
    maxsim,
)


def test_maxsim_handcomputed():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 tokens
    doc_a = np.array([[1.0, 0.0]])           # solo matchea el token 0
    doc_b = np.array([[1.0, 0.0], [0.0, 1.0]])  # matchea ambos tokens
    assert maxsim(q, doc_a) == 1.0
    assert maxsim(q, doc_b) == 2.0
    assert maxsim(q, np.empty((0, 2))) == 0.0


def test_late_interaction_ranking_with_mock():
    # query con 2 tokens; img_b debe ganar (cubre ambos tokens), luego img_a, luego img_c.
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    image_vecs = {
        "img_valid_000000": np.array([[1.0, 0.0]]),               # a -> 1.0
        "img_valid_000001": np.array([[1.0, 0.0], [0.0, 1.0]]),   # b -> 2.0
        "img_valid_000002": np.array([[0.0, 0.2]]),               # c -> 0.2
    }
    text_vecs = {"adufe": q}
    embedder = MockMultiVectorEmbedder(image_vecs, text_vecs)
    retriever = LateInteractionRetriever(embedder, provider=None)

    queries = [Query("q_adufe_en", "adufe", "adufe", "en")]
    ranking = retriever.rank(queries, list(image_vecs), top_k=3)
    order = [d.image_id for d in ranking["q_adufe_en"]]
    assert order == ["img_valid_000001", "img_valid_000000", "img_valid_000002"]
    assert ranking["q_adufe_en"][0].score == 2.0
