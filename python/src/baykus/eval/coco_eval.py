"""pycocotools üzerine ince katman: mAP50, mAP50-95, sınıf ve boyut kırılımları, kırılımlı ölçüm.

Not: maxDets=1000 — DroneVehicle'da görüntü başına 100'den fazla araç olabilir.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETS = 1000
_SIZES = {"small": 1, "medium": 2, "large": 3}  # COCOeval areaRng indeksleri


def _ap(prec: np.ndarray) -> float | None:
    valid = prec[prec > -1]
    return float(valid.mean()) if valid.size else None


def _empty(gt: COCO, n_images: int) -> dict:
    names = [c["name"] for c in gt.loadCats(gt.getCatIds())]
    return {
        "map50": 0.0,
        "map50_95": 0.0,
        "per_class": {n: 0.0 for n in names},
        "per_size": {k: 0.0 for k in _SIZES},
        "n_images": n_images,
        "n_dets": 0,
    }


def evaluate_coco(gt: COCO, dets: list[dict], img_ids: list[int] | None = None) -> dict:
    ids = sorted(img_ids) if img_ids is not None else sorted(gt.getImgIds())
    id_set = set(ids)
    dets = [d for d in dets if d["image_id"] in id_set]
    if not dets or not ids:
        return _empty(gt, len(ids))
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.imgIds = ids
        ev.params.maxDets = [1, 10, MAX_DETS]
        ev.evaluate()
        ev.accumulate()
    p = ev.eval["precision"]  # [T, R, K, A, M]
    m = len(ev.params.maxDets) - 1
    cats = gt.loadCats(ev.params.catIds)
    return {
        "map50": _ap(p[0, :, :, 0, m]) or 0.0,
        "map50_95": _ap(p[:, :, :, 0, m]) or 0.0,
        "per_class": {c["name"]: _ap(p[0, :, k, 0, m]) for k, c in enumerate(cats)},
        "per_size": {s: _ap(p[0, :, :, a, m]) for s, a in _SIZES.items()},
        "n_images": len(ids),
        "n_dets": len(dets),
    }


def evaluate_slices(gt: COCO, dets: list[dict], slices: dict[str, list[int]]) -> dict[str, dict]:
    return {name: evaluate_coco(gt, dets, ids) for name, ids in slices.items()}
