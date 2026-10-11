"""Ultralytics predictions.json → bizim COCO image_id'lerimiz; tahmin dosyası (kanca K3).

Ultralytics image_id'yi dosya adından (int(stem)) üretir; bizim COCO kimliklerimiz sıralı
numaradır ve dışlanan dosyalar yüzünden kayar. Eşleme bu yüzden dosya adı üzerinden yapılır.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from pycocotools.coco import COCO


def remap_to_gt(preds: list[dict], gt: COCO) -> tuple[list[dict], int]:
    by_stem = {Path(im["file_name"]).stem: im["id"] for im in gt.dataset["images"]}
    by_int = {int(s): i for s, i in by_stem.items() if s.isdigit()}
    out, dropped = [], 0
    for p in preds:
        if "file_name" in p:
            img_id = by_stem.get(Path(p["file_name"]).stem)
        elif isinstance(p["image_id"], int):
            img_id = by_int.get(p["image_id"])
        else:
            img_id = by_stem.get(str(p["image_id"]))
        if img_id is None:
            dropped += 1
            continue
        out.append(
            {
                "image_id": img_id,
                "category_id": int(p["category_id"]),
                "bbox": [float(x) for x in p["bbox"]],
                "score": float(p["score"]),
            }
        )
    return out, dropped


def save_predictions(path: Path, preds: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(preds, f)
    return path


def load_predictions(path: Path) -> list[dict]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        return json.load(f)
