"""Referans hareket etiketleri (Faz 4 tasarımı §5.3).

Ürünün ego tahmincisinden bilerek farklı: `t − K` ve `t + K` kareleri **doğrudan** kaydedilir
(SIFT + oran testi + homografi RANSAC, tam çözünürlük, GT kutuları maskeli), pencere ortalanmış. Hız
`v_ref = ‖H(c_{t−K}) − c_{t+K}‖ / (2K) / √(w_{t+K}·h_{t+K})` (pay ve payda `t + K` karesinin
pikselinde). `≤ lo` duran, `≥ hi` hareketli, arası bant; kayıt başarısızsa `unregistered`, iz
`t ± K`'de yoksa `edge`.
IR etiketleri (Faz 3 süzgeci: flag 1, sınıf 1, görünürlük > 0,25).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import cv2
import numpy as np

from baykus.motion.images import Box
from baykus.track.runner import GT_VIS

SIFT_FEATURES = 2000
RATIO = 0.75
RANSAC_PX = 1.5
MASK_PX = 6
MIN_INLIERS = 50

Label = Literal["moving", "stationary", "band", "unregistered", "edge"]


class LabelRow(NamedTuple):
    seq: str
    frame: int
    track_id: int
    v_ref: float | None
    label: Label


class Frames(Protocol):
    name: str
    frames: list[int]

    def ir(self, n: int) -> np.ndarray: ...


def gt_tracks(tracks_csv: Path, seq: str) -> dict[int, dict[int, Box]]:
    """`pack_labels` iz dosyası → iz → kare → kutu (IR, Faz 3 süzgeci)."""
    out: dict[int, dict[int, Box]] = {}
    with Path(tracks_csv).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["seq"] != seq or r.get("modality", "ir") != "ir":
                continue
            if int(r["flag"]) != 1 or int(r["cls"]) != 1 or float(r["vis"]) <= GT_VIS:
                continue
            x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
            out.setdefault(int(r["track_id"]), {})[int(r["frame"])] = (x, y, x + w, y + h)
    return out


def _mask(shape: tuple[int, int], boxes: list[Box], pad: int) -> np.ndarray:
    m = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        a, b = max(0, math.floor(x1) - pad), max(0, math.floor(y1) - pad)
        m[b : math.ceil(y2) + pad, a : math.ceil(x2) + pad] = 0
    return m


_sift = None


def _features(img: np.ndarray, boxes: list[Box]):
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(SIFT_FEATURES)
    return _sift.detectAndCompute(img, _mask(img.shape[:2], boxes, MASK_PX))


def _homography(fa, fb) -> np.ndarray | None:
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    h, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if h is None or inl is None or int(inl.sum()) < MIN_INLIERS:
        return None
    return h


def register_direct(a: np.ndarray, b: np.ndarray, mask_a: list[Box], mask_b: list[Box]):
    """a → b homografi (3×3) ya da None (iç nokta < 50)."""
    return _homography(_features(a, mask_a), _features(b, mask_b))


def _center(b: Box) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def label_sequence(
    frames: Frames, gt: dict[int, dict[int, Box]], k: int = 15, lo: float = 0.03, hi: float = 0.10
) -> list[LabelRow]:
    present = set(frames.frames)
    boxes_at: dict[int, list[Box]] = {}
    for per in gt.values():
        for f, b in per.items():
            boxes_at.setdefault(f, []).append(b)
    feats: dict[int, tuple] = {}
    homs: dict[tuple[int, int], np.ndarray | None] = {}

    def feat(f: int):
        if f not in feats:
            feats[f] = _features(frames.ir(f), boxes_at.get(f, []))
        return feats[f]

    def hom(a: int, b: int):
        if (a, b) not in homs:
            homs[(a, b)] = _homography(feat(a), feat(b))
        return homs[(a, b)]

    rows: list[LabelRow] = []
    for tid in sorted(gt):
        per = gt[tid]
        for t in sorted(per):
            a, b = t - k, t + k
            if a not in per or b not in per or a not in present or b not in present:
                rows.append(LabelRow(frames.name, t, tid, None, "edge"))
                continue
            h = hom(a, b)
            if h is None:
                rows.append(LabelRow(frames.name, t, tid, None, "unregistered"))
                continue
            p = cv2.perspectiveTransform(_center(per[a]).reshape(1, 1, 2), h)[0, 0]
            x1, y1, x2, y2 = per[b]
            size = math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))
            v = float(np.linalg.norm(_center(per[b]) - p)) / (2 * k) / size
            label: Label = "stationary" if v <= lo else "moving" if v >= hi else "band"
            rows.append(LabelRow(frames.name, t, tid, v, label))
    rows.sort(key=lambda r: (r.frame, r.track_id))
    return rows


def write_labels(path: Path, rows: list[LabelRow]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(LabelRow._fields)
        for r in rows:
            w.writerow(
                [r.seq, r.frame, r.track_id, "" if r.v_ref is None else f"{r.v_ref:.6f}", r.label]
            )
    return path


def read_labels(path: Path) -> list[LabelRow]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [
            LabelRow(
                r["seq"],
                int(r["frame"]),
                int(r["track_id"]),
                float(r["v_ref"]) if r["v_ref"] else None,
                r["label"],  # type: ignore[arg-type]
            )
            for r in csv.DictReader(f)
        ]
