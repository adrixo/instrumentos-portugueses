"""Parser de las anotaciones COCO del dataset de instrumentos portugueses.

Hechos verificados del dataset (ver plan):
- 23 categorías COCO, pero la id 0 (`instruments`) es una supercategoría con 0 anotaciones
  → se excluye; quedan 22 instrumentos reales.
- El `image_id` COCO reinicia en 0 en cada split → aquí NO se usa como identificador global;
  la anonimización (ver `anonymize.py`) genera un id con namespace de split.
- Multi-etiqueta: una imagen puede contener varios instrumentos.

Este módulo NO anonimiza: devuelve los datos crudos (incluido `file_name`), que solo consume
la etapa `prepare-data`. El resto del pipeline trabaja sobre artefactos anonimizados.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Nombres de "supercategorías paraguas" que no son instrumentos y deben excluirse del ground truth.
UMBRELLA_CATEGORY_NAMES = frozenset({"instruments"})

# Variantes habituales del nombre de la carpeta de validación.
_SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val", "validation"),
    "test": ("test",),
}


@dataclass(frozen=True)
class CocoImage:
    """Una imagen con los instrumentos (distintos) anotados en ella."""

    coco_id: int
    file_name: str
    width: int
    height: int
    category_names: tuple[str, ...]  # instrumentos distintos presentes, ordenados; paraguas excluido


@dataclass(frozen=True)
class CocoSplit:
    """Resultado de parsear un split COCO."""

    split: str
    images: tuple[CocoImage, ...]
    categories: dict[int, str]  # id COCO -> nombre (solo instrumentos reales)
    excluded_category_ids: tuple[int, ...]
    annotation_path: Path


def find_split_dir(root: Path, split: str) -> Path:
    """Localiza la carpeta del split admitiendo variantes (valid/val/validation)."""
    aliases = _SPLIT_ALIASES.get(split, (split,))
    for alias in aliases:
        candidate = root / alias
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No se encontró la carpeta del split '{split}' en {root} "
        f"(probadas: {', '.join(aliases)})"
    )


def find_annotation_file(split_dir: Path) -> Path:
    """Localiza el JSON de anotaciones COCO dentro de la carpeta del split."""
    preferred = split_dir / "_annotations.coco.json"
    if preferred.is_file():
        return preferred
    # Fallbacks: instances_*.json u otro *.coco.json / *.json
    for pattern in ("instances_*.json", "*.coco.json", "*.json"):
        matches = sorted(split_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No se encontró fichero de anotaciones COCO en {split_dir}")


def _instrument_categories(raw_categories: list[dict]) -> tuple[dict[int, str], tuple[int, ...]]:
    """Separa categorías de instrumentos de las supercategorías paraguas a excluir."""
    kept: dict[int, str] = {}
    excluded: list[int] = []
    for cat in raw_categories:
        name = str(cat["name"]).strip()
        if name.lower() in UMBRELLA_CATEGORY_NAMES:
            excluded.append(int(cat["id"]))
        else:
            kept[int(cat["id"])] = name
    return kept, tuple(sorted(excluded))


def parse_coco(annotation_path: Path, split: str) -> CocoSplit:
    """Parsea un fichero COCO y devuelve imágenes + categorías (instrumentos reales)."""
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))

    categories, excluded_ids = _instrument_categories(data["categories"])
    excluded_set = set(excluded_ids)

    # Acumular, por imagen, el conjunto de categorías de instrumento presentes.
    per_image_cats: dict[int, set[int]] = {}
    for ann in data["annotations"]:
        cat_id = int(ann["category_id"])
        if cat_id in excluded_set:
            continue
        per_image_cats.setdefault(int(ann["image_id"]), set()).add(cat_id)

    images: list[CocoImage] = []
    for img in data["images"]:
        coco_id = int(img["id"])
        cat_ids = per_image_cats.get(coco_id, set())
        names = tuple(sorted(categories[c] for c in cat_ids))
        images.append(
            CocoImage(
                coco_id=coco_id,
                file_name=str(img["file_name"]),
                width=int(img.get("width", 0)),
                height=int(img.get("height", 0)),
                category_names=names,
            )
        )

    images.sort(key=lambda im: im.coco_id)  # orden estable y reproducible
    return CocoSplit(
        split=split,
        images=tuple(images),
        categories=categories,
        excluded_category_ids=excluded_ids,
        annotation_path=Path(annotation_path),
    )


def parse_split(root: Path, split: str) -> CocoSplit:
    """Conveniencia: localiza carpeta + fichero y parsea el split."""
    split_dir = find_split_dir(Path(root), split)
    ann = find_annotation_file(split_dir)
    return parse_coco(ann, split)
