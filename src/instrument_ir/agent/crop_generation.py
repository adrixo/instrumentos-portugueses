"""Generación de crops para el agente B5 (ADR §5). Funciones puras (sin VLM), testeables.

MVP: full_image, center_crop, grid_2x2. Ampliable a saliency/objectness (GroundingDINO/SAM) más tarde.
"""

from __future__ import annotations


def center_crop(image, frac: float = 0.6):
    """Recorte central que cubre `frac` del ancho/alto."""
    w, h = image.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


def grid_2x2(image) -> list:
    """Cuatro cuadrantes de la imagen."""
    w, h = image.size
    mx, my = w // 2, h // 2
    boxes = [(0, 0, mx, my), (mx, 0, w, my), (0, my, mx, h), (mx, my, w, h)]
    return [image.crop(b) for b in boxes]


def generate_crops(image, strategies: list[str], max_crops: int = 5) -> list[tuple[str, object]]:
    """Devuelve [(crop_id, imagen), ...] según las estrategias activas, hasta max_crops."""
    crops: list[tuple[str, object]] = []
    if "center" in strategies:
        crops.append(("center", center_crop(image)))
    if "grid2x2" in strategies:
        for i, c in enumerate(grid_2x2(image)):
            crops.append((f"grid{i}", c))
    crops = crops[:max_crops]
    # Renombrar a crop_NN para encajar con el esquema de traza.
    return [(f"crop_{i:02d}", img) for i, (_, img) in enumerate(crops)]
