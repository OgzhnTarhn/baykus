"""Bir tahmin listesi için spec §12.4 metrik bloğu (FR-V1, V2, V8)."""

from __future__ import annotations

from pycocotools.coco import COCO

from baykus.eval.bootstrap import bootstrap_ci, collect_matches, precision_recall, subset
from baykus.eval.coco_eval import evaluate_coco

ILLUMINATION = ("day", "night", "dark_night")


def build_metrics(
    gt: COCO,
    dets: list[dict],
    illumination: dict[str, list[int]] | None,
    n_boot: int = 1000,
    seed: int = 0,
    workers: int = 1,
) -> dict:
    base = evaluate_coco(gt, dets)
    m = collect_matches(gt, dets)
    precision, recall = precision_recall(m)
    per_ill, ci_ill = {}, {}
    for name in ILLUMINATION:
        ids = (illumination or {}).get(name) or []
        if ids:
            per_ill[name] = evaluate_coco(gt, dets, ids)["map50"]
            ci_ill[name] = list(bootstrap_ci(subset(m, ids), n_boot, seed, workers))
        else:
            per_ill[name], ci_ill[name] = None, [None, None]
    return {
        "map50": base["map50"],
        "map50_95": base["map50_95"],
        "precision": precision,
        "recall": recall,
        "per_class": base["per_class"],
        "per_illumination": per_ill,
        "per_size": base["per_size"],
        "ci95": {"map50": list(bootstrap_ci(m, n_boot, seed, workers)), "per_illumination": ci_ill},
        "n_images": base["n_images"],
        "n_dets": base["n_dets"],
    }
