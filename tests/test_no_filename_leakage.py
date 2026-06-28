"""Test crítico de prevención de fuga (ADR §15.2).

Garantiza que los artefactos PÚBLICOS (corpus, runfile, qrels) no contienen nombres de archivo,
ids de Vimeo ni hashes; que los image_id son anónimos; y que el corpus no expone columnas con labels.
El nombre del instrumento consultado SÍ está permitido (es la query), por eso no es token prohibido.
"""

from __future__ import annotations

import re

import pandas as pd

from instrument_ir.data.prepare_dataset import load_mapping, prepare_dataset
from instrument_ir.data.qrels import build_qrels, write_qrels_trec
from instrument_ir.data.queries import Query
from instrument_ir.retrieval.base import DummyRetriever
from instrument_ir.utils.trec import write_run_trec

ANON_ID_RE = re.compile(r"^img_(train|valid|test)_\d{6}$")
FORBIDDEN_PUBLIC_COLUMNS = {
    "file_name", "vimeo_id", "frame", "rf_hash", "filename_instrument", "gt_instruments",
    "path", "category", "label",
}


def _forbidden_tokens(mapping: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    for col in ("file_name", "vimeo_id", "rf_hash"):
        for v in mapping[col].dropna().astype(str):
            tokens.add(v)
    return {t for t in tokens if t}


def test_public_corpus_has_no_label_columns(coco_root, tmp_path):
    out = tmp_path / "processed"
    prepare_dataset(coco_root, out, splits=("valid",))
    corpus = pd.read_parquet(out / "valid" / "corpus.parquet")
    assert FORBIDDEN_PUBLIC_COLUMNS.isdisjoint(corpus.columns)
    assert set(corpus.columns) == {"image_id", "split", "width", "height"}


def test_image_ids_are_anonymous(coco_root, tmp_path):
    out = tmp_path / "processed"
    prepare_dataset(coco_root, out, splits=("valid",))
    corpus = pd.read_parquet(out / "valid" / "corpus.parquet")
    for iid in corpus["image_id"]:
        assert ANON_ID_RE.match(iid), iid


def test_no_filename_or_vimeo_in_public_artifacts(coco_root, tmp_path):
    out = tmp_path / "processed"
    prepare_dataset(coco_root, out, splits=("valid",))
    mapping = load_mapping(out / "image_id_mapping.parquet", split="valid")
    forbidden = _forbidden_tokens(mapping)
    assert forbidden  # el fixture tiene fuga en los nombres, debe haber tokens

    queries = [Query("q_adufe_en", "adufe", "adufe", "en")]
    qrels_path = tmp_path / "valid.qrels"
    write_qrels_trec(build_qrels(mapping, queries), qrels_path)

    corpus = pd.read_parquet(out / "valid" / "corpus.parquet")
    image_ids = corpus["image_id"].tolist()
    rankings = DummyRetriever().rank(queries, image_ids, top_k=10)
    run_path = tmp_path / "run.trec"
    write_run_trec(rankings, "B1_dummy_valid", run_path)

    # Ninguno de los tokens prohibidos puede aparecer en los artefactos públicos.
    public_blobs = [
        corpus.to_csv(index=False),
        run_path.read_text(),
        qrels_path.read_text(),
    ]
    for blob in public_blobs:
        for token in forbidden:
            assert token not in blob, f"FUGA: '{token}' aparece en artefacto público"
