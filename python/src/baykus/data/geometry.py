"""Kutu geometrisi: poligon → yatay kutu, kırpma/kaydırma, YOLO normalizasyonu."""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]


def polygon_to_hbb(points: np.ndarray) -> Box:
    """(N, 2) ya da düz [x1, y1, x2, y2, ...] noktaları kapsayan eksen hizalı kutu."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        raise ValueError("polygon_to_hbb en az 2 nokta ister")
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def shift_and_clip(
    box: Box, dx: float, dy: float, width: int, height: int, min_size: float = 1.0
) -> Box | None:
    """Kutuyu (-dx, -dy) kaydırıp [0, width] × [0, height]'e kırpar.

    Kırpmadan sonra bir kenarı `min_size`'dan kısa kalırsa (tamamen kenarlıkta kalan ya da
    sıfır alanlı kutu) None döner.
    """
    x1, y1, x2, y2 = box
    x1, x2 = x1 - dx, x2 - dx
    y1, y2 = y1 - dy, y2 - dy
    x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
    y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return float(x1), float(y1), float(x2), float(y2)


def xyxy_to_yolo(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    """Piksel (x1, y1, x2, y2) → YOLO (cx, cy, w, h), görüntü boyutuna oranlı (0-1)."""
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2 / width,
        (y1 + y2) / 2 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )
