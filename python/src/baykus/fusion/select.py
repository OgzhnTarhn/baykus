"""E4 / E4b parametre seçimi (FR-V7): sadece val, IR etiketleri, ölçüt mAP50.
Eşitlikte ızgara sırasındaki ilk hücre (belirlenimci)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pycocotools.coco import COCO

from baykus.eval.coco_eval import evaluate_coco
from baykus.fusion.config import FusionConfig
from baykus.fusion.late import fuse_predictions


def grid_search(
    rgb: list[dict],
    ir: list[dict],
    gt: COCO,
    *,
    iou_thrs: Iterable[float],
    ir_weights: Iterable[float],
    skip_box_thr: float,
    img_ids: list[int] | None = None,
    workers: int = 1,
) -> list[dict]:
    images = {im["id"]: (im["width"], im["height"]) for im in gt.dataset["images"]}
    rows = []
    for iou in iou_thrs:
        for w in ir_weights:
            fused = fuse_predictions(rgb, ir, images, (1.0, float(w)), iou_thr=iou,
                                     skip_box_thr=skip_box_thr, only=img_ids,
                                     workers=workers)  # fmt: skip
            rows.append(
                {
                    "iou_thr": iou,
                    "ir_weight": float(w),
                    "map50": evaluate_coco(gt, fused, img_ids)["map50"],
                }  # fmt: skip
            )
            print(f"iou {iou} · ir {w}: map50 {rows[-1]['map50']:.4f}", flush=True)
    return rows


def best(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("boş ızgara")
    return max(rows, key=lambda r: r["map50"])  # max eşitlikte ilkini döner


def select_constant(rgb, ir, gt: COCO, cfg: FusionConfig, workers: int = 1) -> dict:
    grid = grid_search(rgb, ir, gt, iou_thrs=cfg.iou_thr, ir_weights=cfg.ir_weight,
                       skip_box_thr=cfg.skip_box_thr, workers=workers)  # fmt: skip
    b = best(grid)
    return {"grid": grid, "selected": {"iou_thr": b["iou_thr"], "w_rgb": 1.0,
                                       "ir_weight": b["ir_weight"]}}  # fmt: skip


def select_per_illumination(
    rgb, ir, gt: COCO, cfg: FusionConfig, slices: Mapping[str, list[int]], iou_thr: float,
    workers: int = 1,
) -> dict:  # fmt: skip
    grid, chosen = [], {}
    for level, ids in slices.items():
        if not ids:
            continue
        rows = grid_search(rgb, ir, gt, iou_thrs=[iou_thr], ir_weights=cfg.ir_weight,
                           skip_box_thr=cfg.skip_box_thr, img_ids=list(ids),
                           workers=workers)  # fmt: skip
        grid += [{**r, "slice": level} for r in rows]
        chosen[level] = best(rows)["ir_weight"]
    if not chosen:
        raise ValueError("aydınlatma dilimi yok")
    return {"grid": grid, "selected": {"iou_thr": iou_thr, "w_rgb": 1.0, "ir_weight": chosen}}
