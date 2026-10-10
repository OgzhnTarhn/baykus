"""Yatay kutular için IoU (xyxy)."""

from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4) ve b (M,4) xyxy kutuları arasındaki IoU tablosu, şekil (N,M).

    Birleşimi 0 olan çiftlerde (sıfır alanlı kutular) sonuç 0'dır; NaN üretilmez.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    # (N,1) ile (1,M) broadcasting → her a-b çifti için kesişim kutusunun kenarları (N,M)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    # Örtüşmeyen çiftlerde genişlik/yükseklik negatif çıkar → 0'a kırpılır
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    # Paydada 0 yerine 1 kullanılır ki 0/0 uyarısı ve NaN oluşmasın; o hücreler zaten 0 döner
    safe = np.where(union > 0, union, 1.0)
    return np.where(union > 0, inter / safe, 0.0)
