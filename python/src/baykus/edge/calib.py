"""Kalibrasyon listesi (H5): INT8 için görüntü adları ve Ultralytics veri YAML'ı."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from baykus.classes import CLASS_NAMES
from baykus.data.convert import activate_label_set

EXT = {"rgb": ".jpg", "ir": ".png", "rgbt4": ".tif"}


def calibration_stems(root: Path, split: str = "train", n: int = 500, seed: int = 42) -> list[str]:
    """rgb/images/<split>/*.jpg adları sıralı; seed'li örneklem, dönüş sıralı."""
    root = Path(root)
    names = sorted(p.stem for p in (root / "rgb/images" / split).glob("*.jpg"))
    if not names:
        raise ValueError(f"kalibrasyon görüntüsü yok: {root / 'rgb/images' / split}")
    idx = np.random.default_rng(seed).choice(len(names), size=min(n, len(names)), replace=False)
    return sorted(names[i] for i in idx)


def write_calib_data(root: Path, modality: str, stems: Sequence[str], out_dir: Path) -> Path:
    """calib_<modality>.txt (mutlak yollar) + data_<modality>_calib.yaml. YAML yolunu döndürür."""
    root, out_dir = Path(root), Path(out_dir)
    activate_label_set(root, modality, "ir")
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"calib_{modality}.txt"
    lines = [
        (root / modality / "images/train" / f"{s}{EXT[modality]}").absolute().as_posix()
        for s in stems
    ]
    txt.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    data = {
        "path": (root / modality).as_posix(),
        "train": txt.absolute().as_posix(),
        "val": "images/val",
        "channels": 4 if modality == "rgbt4" else 3,
        "names": list(CLASS_NAMES),
    }
    yml = out_dir / f"data_{modality}_calib.yaml"
    yml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yml
