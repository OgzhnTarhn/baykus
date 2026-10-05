"""RGB + termal → 4 kanallı TIFF (FR-D5). Biçim ADR 0004'te."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

TIFF_LAYOUT = "pages"  # ADR 0004: her kanal ayrı gri sayfa (Spike A)


def stack_rgbt(bgr: np.ndarray, ir: np.ndarray) -> np.ndarray:
    if bgr.shape[:2] != ir.shape[:2]:
        raise ValueError(f"boyut uyuşmazlığı: {bgr.shape[:2]} vs {ir.shape[:2]}")
    ir2 = ir if ir.ndim == 2 else ir[..., 0]
    return np.dstack([bgr[..., 2], bgr[..., 1], bgr[..., 0], ir2]).astype(np.uint8)


def write_rgbt4(path: Path, rgbt: np.ndarray) -> None:
    """Her kanal ayrı gri sayfa. Tek sayfalı RGB+alfa TIFF'te OpenCV, RGB'yi 4. kanalla
    çarpıp BGR'ye çeviriyor (Spike A); çok sayfalıyı Ultralytics sırayla yığıyor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path) as tw:
        for c in range(rgbt.shape[2]):
            tw.write(rgbt[..., c], photometric="minisblack", compression="deflate")
