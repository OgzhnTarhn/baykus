"""Görüntü düzeyinde bootstrap güven aralığı ve eşleştirilmiş fark (FR-V8).

pycocotools evaluate() bir kez çalışır (IoU 0.5, alan 'all', maxDets=MAX_DETS); her yeniden
örneklemede accumulate adımları burada birebir tekrarlanır. Örneklem = kendisi olduğunda sonuç
evaluate_coco()['map50'] ile aynıdır (testli).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from baykus.eval.coco_eval import MAX_DETS

_REC_THRS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class ClassMatches:
    offsets: np.ndarray  # (I+1,) görüntü başına tespit dilimleri
    scores: np.ndarray  # (D,)
    tp: np.ndarray  # (D,) bool: IoU>=0.5 eşleşti ve yok sayılmadı
    fp: np.ndarray  # (D,) bool: eşleşmedi ve yok sayılmadı
    npig: np.ndarray  # (I,) yok sayılmayan GT sayısı


@dataclass(frozen=True)
class Matches:
    img_ids: list[int]
    classes: dict[str, ClassMatches]


def _pack(per_img: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]) -> ClassMatches:
    lens = np.array([len(s) for s, _, _, _ in per_img], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

    def cat(i: int, dtype) -> np.ndarray:
        parts = [np.asarray(x[i], dtype=dtype) for x in per_img]
        return np.concatenate(parts) if parts else np.zeros(0, dtype)

    return ClassMatches(
        offsets=offsets,
        scores=cat(0, np.float64),
        tp=cat(1, bool),
        fp=cat(2, bool),
        npig=np.array([x[3] for x in per_img], dtype=np.int64),
    )


def collect_matches(gt: COCO, dets: list[dict], img_ids: list[int] | None = None) -> Matches:
    ids = sorted(img_ids) if img_ids is not None else sorted(gt.getImgIds())
    id_set = set(ids)
    dets = [d for d in dets if d["image_id"] in id_set]
    cats = gt.loadCats(
        sorted(gt.getCatIds())
    )  # COCOeval p.catIds sırası; evalImgs indeksi buna bağlı
    evals = None
    if dets and ids:
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(dets), iouType="bbox")
            ev.params.imgIds = ids
            ev.params.iouThrs = np.array([0.5])
            ev.params.maxDets = [MAX_DETS]
            ev.params.areaRng = [[0, 1e10]]
            ev.params.areaRngLbl = ["all"]
            ev.evaluate()
        evals = ev.evalImgs
    empty = (np.zeros(0), np.zeros(0, bool), np.zeros(0, bool))
    classes = {}
    for k, c in enumerate(cats):
        per_img = []
        for i, img in enumerate(ids):
            if evals is None:
                n_gt = len(gt.getAnnIds(imgIds=[img], catIds=[c["id"]], iscrowd=False))
                per_img.append((*empty, n_gt))
                continue
            e = evals[k * len(ids) + i]
            if e is None:
                per_img.append((*empty, 0))
                continue
            dtm = np.asarray(e["dtMatches"][0])
            dti = np.asarray(e["dtIgnore"][0]).astype(bool)
            gti = np.asarray(e["gtIgnore"])
            per_img.append(
                (
                    np.asarray(e["dtScores"]),
                    (dtm > 0) & ~dti,
                    (dtm == 0) & ~dti,
                    int((gti == 0).sum()),
                )
            )
        classes[c["name"]] = _pack(per_img)
    return Matches(ids, classes)


def _take(cm: ClassMatches, s: np.ndarray) -> ClassMatches:
    lens = np.diff(cm.offsets)[s]
    starts = cm.offsets[:-1][s]
    total = int(lens.sum())
    if total:
        shift = starts - np.concatenate([[0], np.cumsum(lens)[:-1]])
        idx = np.repeat(shift, lens) + np.arange(total)
    else:
        idx = np.zeros(0, dtype=np.int64)
    return ClassMatches(
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        scores=cm.scores[idx],
        tp=cm.tp[idx],
        fp=cm.fp[idx],
        npig=cm.npig[s],
    )


def subset(m: Matches, img_ids: list[int]) -> Matches:
    pos = {img: i for i, img in enumerate(m.img_ids)}
    keep = sorted(i for i in set(img_ids) if i in pos)
    s = np.array([pos[i] for i in keep], dtype=np.int64)
    return Matches(keep, {name: _take(cm, s) for name, cm in m.classes.items()})


def _ap(cm: ClassMatches) -> float | None:
    npig = int(cm.npig.sum())
    if npig == 0:
        return None  # pycocotools: precision -1 → ortalamaya girmez
    order = np.argsort(-cm.scores, kind="mergesort")
    tp_sum = np.cumsum(cm.tp[order]).astype(float)
    fp_sum = np.cumsum(cm.fp[order]).astype(float)
    q = np.zeros(len(_REC_THRS))
    if len(tp_sum):
        rc = tp_sum / npig
        pr = tp_sum / (fp_sum + tp_sum + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        inds = np.searchsorted(rc, _REC_THRS, side="left")
        ok = inds < len(pr)
        q[ok] = pr[inds[ok]]
    return float(q.mean())


def ap50(m: Matches, sample: np.ndarray | None = None) -> float:
    s = np.arange(len(m.img_ids)) if sample is None else sample
    aps = [a for a in (_ap(_take(cm, s)) for cm in m.classes.values()) if a is not None]
    return float(np.mean(aps)) if aps else 0.0


def bootstrap_ci(m: Matches, n: int = 1000, seed: int = 0) -> tuple[float | None, float | None]:
    k = len(m.img_ids)
    if k == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    vals = np.array([ap50(m, rng.integers(0, k, k)) for _ in range(n)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_diff_ci(a: Matches, b: Matches, n: int = 1000, seed: int = 0) -> dict:
    if a.img_ids != b.img_ids:
        raise ValueError("eşleştirilmiş bootstrap için iki model aynı görüntü listesinde ölçülmeli")
    k = len(a.img_ids)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = rng.integers(0, k, k)
        diffs.append(ap50(a, s) - ap50(b, s))
    lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    return {"diff": ap50(a) - ap50(b), "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}


def precision_recall(m: Matches, conf: float = 0.25) -> tuple[float | None, float | None]:
    tp = fp = npig = 0
    for cm in m.classes.values():
        keep = cm.scores >= conf
        tp += int(cm.tp[keep].sum())
        fp += int(cm.fp[keep].sum())
        npig += int(cm.npig.sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / npig if npig else None
    return precision, recall
