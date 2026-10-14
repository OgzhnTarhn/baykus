"""Uçtan uca (slow) testler için minik veri seti: bizim işlenmiş düzenimizle aynı."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baykus.data.coco import yolo_to_coco
from baykus.data.yamls import write_data_yamls
from baykus.io import imwrite_any

SIZE = 64


def make_synthetic_dataset(
    root: Path, n_train: int = 8, n_val: int = 4, with_ir: bool = False
) -> Path:
    """rgb/images/{train,val}, labels_{ir,rgb}/{train,val}, data_*.yaml, coco/, meta/ üretir.
    Görüntü adları 00001.. biçiminde; val'de bir numara bilerek atlanır (dışlanmış dosya).
    with_ir=True: ir/images/{train,val}/*.png (gri) de yazılır."""
    root = Path(root)
    rng = np.random.default_rng(0)
    stems = {
        "train": [f"{i:05d}" for i in range(1, n_train + 1)],
        "val": [f"{i:05d}" for i in range(101, 101 + n_val + 1) if i != 102],
    }
    for split, names in stems.items():
        for s in names:
            img = rng.integers(0, 60, (SIZE, SIZE, 3), dtype=np.uint8)
            img[20:40, 16:48] = 255
            imwrite_any(root / "rgb/images" / split / f"{s}.jpg", img)
            if with_ir:
                ir = rng.integers(0, 60, (SIZE, SIZE), dtype=np.uint8)
                ir[20:40, 16:48] = 200
                imwrite_any(root / "ir/images" / split / f"{s}.png", ir)
            for ls in ("ir", "rgb"):
                p = root / f"labels_{ls}" / split / f"{s}.txt"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("0 0.5 0.46875 0.5 0.3125\n", encoding="utf-8")
    write_data_yamls(root)
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for ls in ("ir", "rgb"):
        coco = yolo_to_coco(root / f"labels_{ls}" / "val", stems["val"], SIZE, SIZE)
        (root / "coco").mkdir(exist_ok=True)
        (root / "coco" / f"val_{ls}.json").write_text(json.dumps(coco), encoding="utf-8")
    ids = list(range(1, len(stems["val"]) + 1))
    slices = {"illumination": {"day": ids[:2], "night": ids[2:], "dark_night": []}}
    (meta / "slices_val.json").write_text(json.dumps(slices), encoding="utf-8")
    return root
