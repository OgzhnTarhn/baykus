"""Bir bozulma koşulu için değerlendirme klasörü (FR-V4, kancalar K1-K2).

Düzen işlenmiş verinin aynısı: <out>/{rgb,ir,rgbt4}/images/<split>, labels_<set>/<split>,
coco/<split>_<set>.json, data_*.yaml. Böylece baykus-eval aynı yoldan ölçer.
Bozulmuş RGB PNG yazılır: JPEG yeniden sıkıştırması kendi bozulmasını eklerdi.
"""

from __future__ import annotations

import multiprocessing
import shutil
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from baykus.data.multichannel import stack_rgbt, write_rgbt4
from baykus.data.yamls import write_data_yamls
from baykus.io import imread_any, imwrite_any
from baykus.robustness.corruptions import Condition, apply_condition

MODALITIES = ("rgb", "ir", "rgbt4")


def _read(path: Path, flag: int) -> np.ndarray:
    img = imread_any(path, flag)
    if img is None:
        raise OSError(f"okunamadı: {path}")
    return img


def _write_one(job: tuple) -> None:
    rgb_path, ir_path, out, split, stem, cond, seed, mods = job
    need_rgb = bool({"rgb", "rgbt4"} & set(mods))
    need_ir = bool({"ir", "rgbt4"} & set(mods))
    bgr = _read(rgb_path, cv2.IMREAD_COLOR) if need_rgb else None
    ir = _read(ir_path, cv2.IMREAD_GRAYSCALE) if need_ir else None
    # apply_condition iki modaliteyi de ister; okunmayan tarafın yerine boş görüntü (yazılmaz)
    if bgr is None:
        bgr = np.zeros((*ir.shape, 3), np.uint8)
    if ir is None:
        ir = np.zeros(bgr.shape[:2], np.uint8)
    bgr, ir = apply_condition(bgr, ir, cond, seed=seed, stem=stem)
    if "rgb" in mods:
        imwrite_any(out / "rgb/images" / split / f"{stem}.png", bgr)
    if "ir" in mods:
        imwrite_any(out / "ir/images" / split / f"{stem}.png", ir)
    if "rgbt4" in mods:
        write_rgbt4(out / "rgbt4/images" / split / f"{stem}.tif", stack_rgbt(bgr, ir))


def make_condition(
    src: Path,
    out: Path,
    split: str,
    cond: Condition,
    modalities: Iterable[str],
    seed: int,
    workers: int = 1,
    label_set: str = "ir",
) -> int:
    src, out = Path(src), Path(out)
    mods = tuple(sorted(set(modalities)))
    unknown = [m for m in mods if m not in MODALITIES]
    if unknown:
        raise ValueError(f"bilinmeyen modalite: {unknown}; seçenekler: {MODALITIES}")
    stems = sorted(p.stem for p in (src / "rgb/images" / split).glob("*.jpg"))
    jobs = [
        (
            src / "rgb/images" / split / f"{s}.jpg",
            src / "ir/images" / split / f"{s}.png",
            out,
            split,
            s,
            cond,
            seed,
            mods,
        )
        for s in stems
    ]
    if workers <= 1:
        for job in jobs:
            _write_one(job)
    else:
        # torch/CUDA başlamış süreçte (sağlamlık paketi) fork güvensiz
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            list(ex.map(_write_one, jobs, chunksize=32))
    labels = src / f"labels_{label_set}" / split
    if labels.is_dir():
        shutil.copytree(labels, out / f"labels_{label_set}" / split, dirs_exist_ok=True)
    coco = src / "coco" / f"{split}_{label_set}.json"
    if coco.exists():
        (out / "coco").mkdir(parents=True, exist_ok=True)
        shutil.copy2(coco, out / "coco" / coco.name)
    write_data_yamls(out)
    return len(stems)
