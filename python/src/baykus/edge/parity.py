"""İki tahmin listesinin eşliği (H5; FR-E6 toleransları): Colab'da ölçülen doğruluk laptop
çıktısını temsil ediyor mu? H6'da C++ motoru da aynı ölçütle karşılaştırılır."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _xyxy(bbox: list[float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def compare_dets(
    a: list[dict],
    b: list[dict],
    *,
    conf: float = 0.25,
    iou_min: float = 0.99,
    score_tol: float = 1e-3,
) -> dict:
    """score ≥ conf kutular; görüntü ve sınıf başına IoU'ya göre açgözlü eşleme."""
    ga: dict[tuple, list[dict]] = defaultdict(list)
    gb: dict[tuple, list[dict]] = defaultdict(list)
    for src, dst in ((a, ga), (b, gb)):
        for d in src:
            if d["score"] >= conf:
                dst[(d["image_id"], d["category_id"])].append(d)
    count_a = sum(map(len, ga.values()))
    count_b = sum(map(len, gb.values()))
    ious: list[float] = []
    diffs: list[float] = []
    for key in ga.keys() & gb.keys():
        xa, xb = ga[key], gb[key]
        m = np.array([[_iou(_xyxy(p["bbox"]), _xyxy(q["bbox"])) for q in xb] for p in xa])
        while m.size and m.max() > 0:
            i, j = np.unravel_index(int(m.argmax()), m.shape)
            ious.append(float(m[i, j]))
            diffs.append(abs(xa[i]["score"] - xb[j]["score"]))
            m[i, :], m[:, j] = -1, -1
    matched = len(ious)
    min_iou = min(ious) if ious else None
    max_diff = max(diffs) if diffs else None
    ok = (
        count_a == count_b == matched
        and (min_iou is None or min_iou >= iou_min)
        and (max_diff is None or max_diff <= score_tol)
    )
    return {
        "count_a": count_a,
        "count_b": count_b,
        "matched": matched,
        "min_iou": min_iou,
        "max_score_diff": max_diff,
        "ok": bool(ok),
    }
