"""YOLO etiketleri → COCO JSON + kırılım listeleri (FR-D6)."""

from __future__ import annotations

import csv
from pathlib import Path

from baykus.classes import CLASS_NAMES, coco_category_id


def yolo_to_coco(label_dir: Path, stems: list[str], width: int, height: int) -> dict:
    images, anns = [], []
    for img_id, stem in enumerate(sorted(stems), start=1):
        images.append({"id": img_id, "file_name": stem, "width": width, "height": height})
        f = Path(label_dir) / f"{stem}.txt"
        text = f.read_text(encoding="utf-8") if f.exists() else ""
        for line in text.splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            bw, bh = float(w) * width, float(h) * height
            x, y = float(cx) * width - bw / 2, float(cy) * height - bh / 2
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": img_id,
                    "category_id": coco_category_id(int(c)),
                    "bbox": [round(x, 4), round(y, 4), round(bw, 4), round(bh, 4)],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
    cats = [{"id": coco_category_id(i), "name": n} for i, n in enumerate(CLASS_NAMES)]
    return {"images": images, "annotations": anns, "categories": cats}


def illumination_slices(
    csv_path: Path, split: str, stem_to_id: dict[str, int]
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"day": [], "night": [], "dark_night": []}
    with Path(csv_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split and row["stem"] in stem_to_id:
                out[row["illumination"]].append(stem_to_id[row["stem"]])
    for v in out.values():
        v.sort()
    return out
