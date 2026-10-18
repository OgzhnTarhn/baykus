"""Kodlanmış RGB + IR baytlarından bellek görüntüsü (motorun `image_io` modülünün referansı).

RGB `IMREAD_COLOR` → BGR uint8 (H, W, 3); IR `IMREAD_GRAYSCALE` → uint8 (H, W). 3 kanallı ya da
16 bit IR PNG de gri 8 bite iner (M3OT ham IR 3 kanal PNG). Boyutlar aynı olmalı.
"""

from __future__ import annotations

import cv2
import numpy as np


def _decode(data: bytes, flag: int, name: str) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flag) if data else None
    if img is None:
        raise ValueError(f"{name} görüntüsü çözülemedi")
    return img


def decode_pair(rgb_bytes: bytes, ir_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    bgr = _decode(rgb_bytes, cv2.IMREAD_COLOR, "RGB")
    ir = _decode(ir_bytes, cv2.IMREAD_GRAYSCALE, "IR")
    if bgr.shape[:2] != ir.shape[:2]:
        raise ValueError(f"RGB ve IR boyutu farklı: {bgr.shape[:2]} vs {ir.shape[:2]}")
    return bgr, ir
