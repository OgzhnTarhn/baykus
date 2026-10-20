"""Dizi okuma ve yarı çözünürlük gri (Faz 4 tasarımı §5.1; plan Global Constraints).

Yarı çözünürlük sağlık modülünün küçültmesiyle aynı (`baykus.health.features._edges`): RGB önce
griye, sonra `INTER_AREA` ile `(w // 2, h // 2)`. Yarı çözünürlük piksel `i` tam çözünürlükte
`2i + 0,5`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from baykus.engine_ref.cli import read_index
from baykus.io import imread_any

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 (tam çözünürlük pikseli)


def half_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class SequenceFrames:
    """`<dizi>/{rgb/NNNNNN.jpg, ir/NNNNNN.png, index.json}` (demo paketi düzeni).

    Kareler tembel okunur."""

    root: Path
    name: str
    split: str
    frames: list[int]

    def ir(self, n: int) -> np.ndarray:
        return self._read("ir", f"{n:06d}.png", cv2.IMREAD_GRAYSCALE)

    def rgb(self, n: int) -> np.ndarray:
        return self._read("rgb", f"{n:06d}.jpg", cv2.IMREAD_COLOR)

    def _read(self, mod: str, name: str, flags: int) -> np.ndarray:
        img = imread_any(self.root / mod / name, flags)
        if img is None:
            raise FileNotFoundError(f"{self.root / mod / name} okunamadı")
        return img


def load_sequence(seq_dir: Path, allow_test: bool) -> SequenceFrames:
    """Test dizisi yalnız `allow_test` ile (Faz 3 `read_index` kalıbı; aksi hâlde ValueError)."""
    index = read_index(Path(seq_dir), allow_test)
    return SequenceFrames(Path(seq_dir), index["name"], index["split"], list(index["frames"]))
