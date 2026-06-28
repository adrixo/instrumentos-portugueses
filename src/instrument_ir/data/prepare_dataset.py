"""Etapa `prepare-data` (ADR §5): parsea COCO, anonimiza y produce los artefactos.

Produce:
  data/processed/image_id_mapping.parquet      # PRIVADO: image_id ↔ (vimeo_id, file_name, GT)
  data/processed/{split}/corpus.parquet        # PÚBLICO: image_id, width, height (SIN labels/filename)

El corpus público es lo único que consume el retrieval/rerank. Toda señal de etiqueta vive solo en
el mapping privado, que se usa para cargar píxeles y construir qrels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .anonymize import build_mapping
from .coco_parser import parse_split

DEFAULT_SPLITS = ("train", "valid", "test")

# Columnas del corpus PÚBLICO. Deliberadamente NO incluye file_name, vimeo_id ni gt_*.
PUBLIC_CORPUS_COLUMNS = ["image_id", "split", "width", "height"]


@dataclass
class PrepareResult:
    mapping_path: Path
    corpus_paths: dict[str, Path]
    n_images: dict[str, int]
    categories: dict[str, str]  # instrument_id -> instrument_id (las 22 clases reales)


def prepare_dataset(
    raw_root: Path, out_dir: Path, splits: tuple[str, ...] = DEFAULT_SPLITS
) -> PrepareResult:
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mappings: list[pd.DataFrame] = []
    corpus_paths: dict[str, Path] = {}
    n_images: dict[str, int] = {}
    categories: dict[str, str] = {}

    for split in splits:
        coco = parse_split(raw_root, split)
        for name in coco.categories.values():
            categories[name] = name

        mapping = build_mapping(coco)
        mappings.append(mapping)

        # Corpus público: solo columnas seguras.
        corpus = mapping[["image_id", "split", "width", "height"]].copy()
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = split_dir / "corpus.parquet"
        corpus.to_parquet(corpus_path, index=False)
        corpus_paths[split] = corpus_path
        n_images[split] = len(corpus)

    full_mapping = pd.concat(mappings, ignore_index=True)
    mapping_path = out_dir / "image_id_mapping.parquet"
    full_mapping.to_parquet(mapping_path, index=False)

    return PrepareResult(
        mapping_path=mapping_path,
        corpus_paths=corpus_paths,
        n_images=n_images,
        categories=categories,
    )


def load_mapping(mapping_path: Path, split: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(mapping_path)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df
