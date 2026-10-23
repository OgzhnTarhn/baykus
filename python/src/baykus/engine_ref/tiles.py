"""Parçalı çıkarım (Faz 4 tasarımı §3 C): 2 × 2 pencere, kutuları kareye taşıma, sınıfsız NMS.

Pencere boyu karenin 9/16'sı (640 × 512 → 360 × 288); sıra sol üst, sağ üst, sol alt, sağ alt.
Birleştirme: her parçanın `Det`'leri (x + x0, y + y0) ile kareye taşınır; adaylar ham skor azalan,
parça sırası artan, ORT satır indeksi artan sırada açgözlü sınıfsız NMS'ten geçer (IoU > 0,5
bastırılır). Tutulanların `index`'i tutulma sırasıdır (0, 1, ...); dönüş `pipeline.order` sırası.
"""

from __future__ import annotations

from dataclasses import replace

from baykus.engine_ref.pipeline import Det, order

NMS_IOU = 0.5


def windows(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """(x, y, pw, ph) pencereleri; pw = ceil(w·9/16), ph = ceil(h·9/16) (tam sayı aritmetiği)."""
    pw = -(-w * 9 // 16)
    ph = -(-h * 9 // 16)
    return [(0, 0, pw, ph), (w - pw, 0, pw, ph), (0, h - ph, pw, ph), (w - pw, h - ph, pw, ph)]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def merge(per_tile: list[list[Det]], wins: list[tuple[int, int, int, int]]) -> list[Det]:
    """Parça tespitlerini kare koordinatına taşıyıp sınıfsız NMS ile birleştirir."""
    cands: list[tuple[int, int, Det]] = []
    for t, (win, dets) in enumerate(zip(wins, per_tile, strict=True)):
        x0, y0 = win[0], win[1]
        for d in dets:
            x1, y1, x2, y2 = d.box
            cands.append((t, d.index, replace(d, box=(x1 + x0, y1 + y0, x2 + x0, y2 + y0))))
    cands.sort(key=lambda c: (-c[2].score_raw, c[0], c[1]))
    kept: list[Det] = []
    for _t, _i, d in cands:
        if any(_iou(d.box, k.box) > NMS_IOU for k in kept):
            continue
        kept.append(replace(d, index=len(kept)))
    return order(kept)
