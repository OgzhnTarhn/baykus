"""Geç füzyon (FR-V7, kanca K4): iki modelin COCO tahminleri → görüntü görüntü WBF.

Ağırlıklar görüntü başına dışarıdan verilir: {image_id: (w_rgb, w_ir)} ya da bütün görüntüler
için tek (w_rgb, w_ir). E4 (sabit) ve E4b (aydınlatma dilimine göre) bunun özel hâlleri; Faz 2'nin
kare başına uyarlanan ağırlığı da aynı arayüzü kullanır.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from baykus.data.illumination import assign
from baykus.fusion.wbf import fuse_image
from baykus.robustness.corruptions import Condition

Weights = tuple[float, float] | Mapping[int, tuple[float, float]]
CHUNK = 64


def _group(dets: Iterable[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for x in dets:
        out[x["image_id"]].append(x)
    return out


def _fuse_chunk(job: tuple) -> list[dict]:
    items, iou_thr, skip_box_thr = job
    out = []
    for img_id, (width, height), wts, a, b in items:
        fused = fuse_image(
            [a, b], wts, width=width, height=height, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        out += [{"image_id": img_id, **x} for x in fused]
    return out


def fuse_predictions(
    rgb: list[dict],
    ir: list[dict],
    images: Mapping[int, tuple[int, int]],
    weights: Weights,
    *,
    iou_thr: float,
    skip_box_thr: float = 0.001,
    only: Iterable[int] | None = None,
    workers: int = 1,
) -> list[dict]:
    by_rgb, by_ir = _group(rgb), _group(ir)
    unknown = sorted((set(by_rgb) | set(by_ir)) - set(images))
    if unknown:
        raise ValueError(
            f"GT'de olmayan image_id: {unknown[:5]} (tahmin dosyası başka bölümün mü?)"
        )
    ids = sorted(images) if only is None else sorted(set(only))
    missing = [i for i in ids if i not in images]
    if missing:
        raise ValueError(f"GT'de olmayan image_id: {missing[:5]}")
    items = []
    for i in ids:
        a, b = by_rgb.get(i, []), by_ir.get(i, [])
        if not a and not b:
            continue
        if isinstance(weights, Mapping):
            if i not in weights:
                raise KeyError(f"image_id {i} için ağırlık yok")
            wts = weights[i]
        else:
            wts = weights
        items.append((i, images[i], tuple(wts), a, b))
    jobs = [(items[k : k + CHUNK], iou_thr, skip_box_thr) for k in range(0, len(items), CHUNK)]
    if workers <= 1:
        parts = [_fuse_chunk(j) for j in jobs]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            parts = list(ex.map(_fuse_chunk, jobs))
    return [x for p in parts for x in p]


def levels_from_slices(slices: Mapping[str, Iterable[int]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for level, ids in slices.items():
        for i in ids:
            if i in out:
                raise ValueError(f"image_id {i} iki aydınlatma diliminde: {out[i]}, {level}")
            out[i] = level
    return out


def illumination_weights(
    levels: Mapping[int, str], ir_weight: Mapping[str, float], w_rgb: float = 1.0
) -> dict[int, tuple[float, float]]:
    missing = sorted(set(levels.values()) - set(ir_weight))
    if missing:
        raise ValueError(f"şu aydınlatma dilimleri için IR ağırlığı yok: {missing}")
    return {i: (w_rgb, float(ir_weight[lv])) for i, lv in levels.items()}


def condition_levels(
    brightness: Mapping[int, float], thresholds: tuple[float, float], cond: Condition
) -> dict[int, str]:
    """Bozulmuş RGB'nin aydınlatma etiketi (FR-V7: çıkarımda RGB'den türetilir).

    Parlaklık (HSV V ortalaması): karartmada çarpanla ölçeklenir (yuvarlama ±0,5), RGB kaybında
    0; bulanıklık, gürültü ve hizasızlıkta değişmez kabul edilir (Görev 8 / Spec yorumları 5).
    """
    if cond.corruption == "darken":
        f = cond.severity
    elif cond.corruption == "drop_rgb":
        f = 0.0
    else:
        f = 1.0
    ids = sorted(brightness)
    labels = assign(np.array([brightness[i] * f for i in ids]), thresholds)
    return dict(zip(ids, labels, strict=True))


def load_brightness(
    meta: Path, split: str, gt_images: list[dict]
) -> tuple[dict[int, float], tuple[float, float]]:
    meta = Path(meta)
    by_stem = {}
    with (meta / "illumination.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                by_stem[row["stem"]] = float(row["brightness"])
    out = {}
    for im in gt_images:
        stem = Path(im["file_name"]).stem
        if stem not in by_stem:
            raise ValueError(f"{stem}: illumination.csv'de {split} parlaklığı yok")
        out[im["id"]] = by_stem[stem]
    th = json.loads((meta / "illumination_thresholds.json").read_text(encoding="utf-8"))
    return out, (float(th["t_dark"]), float(th["t_day"]))
