"""Ham DroneVehicle klasör düzeni (Spike B ile doğrulanır)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")
RAW_SIZE = (840, 712)  # (w, h)
CROP = 100
OUT_SIZE = (640, 512)  # (w, h)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class SplitDirs:
    rgb: Path
    ir: Path
    label_rgb: Path
    label_ir: Path


def split_dirs(raw_root: Path, split: str) -> SplitDirs:
    base = Path(raw_root) / split
    return SplitDirs(
        rgb=base / f"{split}img",
        ir=base / f"{split}imgr",
        label_rgb=base / f"{split}label",
        label_ir=base / f"{split}labelr",
    )


def find_image(folder: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None
