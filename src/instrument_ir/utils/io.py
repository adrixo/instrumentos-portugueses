"""IO de imágenes vía el mapping privado.

Resolver `image_id -> ruta de píxeles` es el ÚNICO uso permitido del mapping privado en el camino de
inferencia (cargar la imagen). El nombre de archivo nunca se expone al modelo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ImageProvider:
    """Resuelve image_id -> ruta absoluta y carga píxeles, ocultando el nombre de archivo."""

    def __init__(self, mapping: pd.DataFrame, raw_root: Path):
        self._raw_root = Path(raw_root)
        # image_id -> (split, file_name) — privado, no se expone hacia arriba.
        self._index: dict[str, tuple[str, str]] = {
            row.image_id: (row.split, row.file_name)
            for row in mapping.itertuples(index=False)
        }

    def image_ids(self) -> list[str]:
        return list(self._index.keys())

    def path(self, image_id: str) -> Path:
        split, file_name = self._index[image_id]
        return self._raw_root / split / file_name

    def load(self, image_id: str):
        from PIL import Image

        return Image.open(self.path(image_id)).convert("RGB")
