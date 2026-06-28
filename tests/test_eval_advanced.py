from instrument_ir.evaluation.rerank_metrics import (
    candidate_recall_at_n,
    oracle_recall_at_k,
    rerank_gain_at_k,
)
from instrument_ir.evaluation.statistical_tests import (
    bootstrap_ci,
    holm_bonferroni,
    paired_permutation_test,
)
from instrument_ir.reporting.tables import gain_table_md

QRELS = {"q": {"a": 1, "c": 1}}
DENSE = {"q": {"a": 0.9, "b": 0.8, "c": 0.1}}
RERANK = {"q": {"c": 0.9, "a": 0.8, "b": 0.1}}


def test_candidate_and_oracle_recall():
    assert candidate_recall_at_n(DENSE, QRELS, n=2)["macro"] == 0.5  # top2=[a,b], rel∩={a}
    assert oracle_recall_at_k(DENSE, QRELS, n=2, k=10)["macro"] == 0.5
    assert candidate_recall_at_n(DENSE, QRELS, n=3)["macro"] == 1.0  # ambos en top3


def test_rerank_gain():
    # reranked sube c y a al top2 -> recall@2=1.0 vs dense 0.5 -> gain 0.5
    assert rerank_gain_at_k(RERANK, DENSE, QRELS, k=2)["macro"] == 0.5


def test_bootstrap_and_permutation():
    ci = bootstrap_ci([0.5] * 20)
    assert ci["mean"] == 0.5 and ci["lo"] == 0.5 and ci["hi"] == 0.5
    # idénticos -> sin diferencia -> p alto
    assert paired_permutation_test([0.4, 0.6, 0.5], [0.4, 0.6, 0.5]) == 1.0
    # A claramente mayor que B -> p pequeño
    p = paired_permutation_test([1.0] * 12, [0.0] * 12)
    assert p < 0.05


def test_holm_bonferroni():
    adj = holm_bonferroni({"x": 0.001, "y": 0.5})
    assert adj["x"]["significant"] is True
    assert adj["y"]["significant"] is False


def test_gain_table_md():
    all_metrics = {
        "B1": {"per_query": {"q1": {"recall@100": 0.2}, "q2": {"recall@100": 0.3}}},
        "B4": {"per_query": {"q1": {"recall@100": 0.5}, "q2": {"recall@100": 0.6}}},
    }
    table = gain_table_md(all_metrics, [("B1", "B4")], metric="recall@100")
    assert "B4 vs B1" in table
    assert "delta_recall@100" in table
