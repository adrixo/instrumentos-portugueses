"""Fixtures sintéticos: un mini-dataset COCO con la misma forma que el real.

Incluye a propósito:
- la supercategoría paraguas id 0 `instruments` (debe excluirse),
- una imagen multi-etiqueta,
- nombres de archivo con fuga (vimeo_id + instrumento), igual que el dataset real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def coco_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    valid = root / "valid"
    valid.mkdir(parents=True)

    coco = {
        "categories": [
            {"id": 0, "name": "instruments"},  # paraguas, sin anotaciones
            {"id": 1, "name": "adufe"},
            {"id": 2, "name": "cavaquinho"},
        ],
        "images": [
            {"id": 0, "file_name": "100200300_Adufe_25_jpg.rf.deadbeef.jpg", "width": 640, "height": 480},
            {"id": 1, "file_name": "100200300_Cavaquinho_35_jpg.rf.cafebabe.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "999888777_Adufe_50_jpg.rf.0badf00d.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]},
            {"id": 2, "image_id": 0, "category_id": 2, "bbox": [0, 0, 1, 1]},  # img0 multi-etiqueta
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1]},
            {"id": 4, "image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1]},
        ],
    }
    (valid / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return root
