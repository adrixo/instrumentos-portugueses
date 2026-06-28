"""Anonimización de imágenes (ADR §5.2).

El nombre de archivo del dataset contiene el instrumento y el id de Vimeo, p.ej.:

    100204270_Guitarra-Portuguesa_25_jpg.rf.5e6232....jpg
    └─vimeo──┘ └──instrumento───┘ └frame┘     └──hash──┘

Eso es fuga directa. Por eso:
- Se genera un `image_id` anónimo CON namespace de split (`img_{split}_{NNNNNN}`), porque los
  `image_id` COCO colisionan entre splits.
- Se guarda un MAPPING PRIVADO `image_id ↔ {vimeo_id, frame, file_name, gt_instruments}` que SOLO se
  usa para (a) cargar píxeles y (b) mostrar resultados en el buscador (vimeo_id), NUNCA en inferencia.

El ground truth autoritativo son las categorías COCO (`coco_parser`), no el instrumento del nombre;
`filename_instrument` se conserva en el mapping solo como metadato privado de verificación.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .coco_parser import CocoSplit

# {vimeo}_{instrumento}_{frame}_jpg.rf.{hash}.jpg
_FILENAME_RE = re.compile(
    r"^(?P<vimeo>\d+)_(?P<instrument>.+?)_(?P<frame>\d+)_jpg\.rf\.(?P<hash>[0-9A-Za-z]+)\.jpg$"
)


def parse_filename(file_name: str) -> dict[str, str | None]:
    """Extrae vimeo_id / instrumento / frame / hash del nombre. Robusto a formatos inesperados."""
    m = _FILENAME_RE.match(file_name)
    if m:
        return {
            "vimeo_id": m.group("vimeo"),
            "filename_instrument": m.group("instrument").replace("-", " ").lower(),
            "frame": m.group("frame"),
            "rf_hash": m.group("hash"),
        }
    # Fallback: al menos intentar el id de Vimeo (primer bloque numérico).
    lead = re.match(r"^(\d+)", file_name)
    return {
        "vimeo_id": lead.group(1) if lead else None,
        "filename_instrument": None,
        "frame": None,
        "rf_hash": None,
    }


def make_image_id(split: str, index: int) -> str:
    """`img_{split}_{NNNNNN}` — id anónimo estable con namespace de split."""
    return f"img_{split}_{index:06d}"


def build_mapping(coco_split: CocoSplit) -> pd.DataFrame:
    """Construye el DataFrame del mapping privado para un split.

    Columnas:
      image_id, split, coco_id, file_name, vimeo_id, frame, rf_hash,
      filename_instrument, width, height, gt_instruments (lista de instrumentos del GT COCO)
    """
    rows: list[dict] = []
    for index, img in enumerate(coco_split.images):  # imágenes ya ordenadas por coco_id
        parts = parse_filename(img.file_name)
        rows.append(
            {
                "image_id": make_image_id(coco_split.split, index),
                "split": coco_split.split,
                "coco_id": img.coco_id,
                "file_name": img.file_name,
                "vimeo_id": parts["vimeo_id"],
                "frame": parts["frame"],
                "rf_hash": parts["rf_hash"],
                "filename_instrument": parts["filename_instrument"],
                "width": img.width,
                "height": img.height,
                "gt_instruments": list(img.category_names),
            }
        )
    return pd.DataFrame(rows)


def write_mapping(mapping: pd.DataFrame, out_path: Path) -> Path:
    """Escribe el mapping privado en parquet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(out_path, index=False)
    return out_path
