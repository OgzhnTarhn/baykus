"""Windows'ta ASCII dışı yollarda da çalışan görüntü okuma/yazma."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_any(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_any(path: Path, img: np.ndarray, params: list[int] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img, params or [])
    if not ok:
        raise OSError(f"görüntü kodlanamadı: {path}")
    enc.tofile(str(path))
