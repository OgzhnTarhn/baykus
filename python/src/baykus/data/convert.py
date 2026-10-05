"""Kenarlık kırpma + OBB→HBB→YOLO etiket dönüşümü (FR-D2, FR-D3)."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from baykus.classes import class_id
from baykus.data.dronevehicle import parse_annotation
from baykus.data.geometry import polygon_to_hbb, shift_and_clip, xyxy_to_yolo
from baykus.data.layout import CROP, OUT_SIZE, SPLITS, SplitDirs, find_image
from baykus.io import imread_any, imwrite_any

W, H = OUT_SIZE


def to_single_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if (
        img.shape[2] == 3
        and (img[..., 0] == img[..., 1]).all()
        and (img[..., 1] == img[..., 2]).all()
    ):
        return img[..., 0].copy()
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)


def _crop(img: np.ndarray) -> np.ndarray:
    return img[CROP : CROP + H, CROP : CROP + W]


def convert_labels(xml_path: Path) -> tuple[list[str], dict]:
    res = parse_annotation(xml_path)
    lines: list[str] = []
    dropped = 0
    for obj in res.objects:
        box = shift_and_clip(polygon_to_hbb(obj.points), CROP, CROP, W, H)
        if box is None:
            dropped += 1
            continue
        cx, cy, bw, bh = xyxy_to_yolo(box, W, H)
        lines.append(f"{class_id(obj.cls_name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    stats = {
        "kept": len(lines),
        "dropped": dropped,
        "unknown": len(res.unknown_classes),
        "skipped": res.skipped_shapes,
    }
    return lines, stats


def convert_split(dirs: SplitDirs, split: str, stems: list[str], out_root: Path) -> dict:
    total = {
        f"{s}_{k}": 0 for s in ("ir", "rgb") for k in ("kept", "dropped", "unknown", "skipped")
    }
    for stem in tqdm(stems, desc=f"convert {split}", leave=False):
        rgb = imread_any(find_image(dirs.rgb, stem), cv2.IMREAD_COLOR)
        ir = imread_any(find_image(dirs.ir, stem), cv2.IMREAD_UNCHANGED)
        imwrite_any(
            out_root / "rgb/images" / split / f"{stem}.jpg",
            _crop(rgb),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        imwrite_any(out_root / "ir/images" / split / f"{stem}.png", _crop(to_single_channel(ir)))
        for label_set, folder in (("ir", dirs.label_ir), ("rgb", dirs.label_rgb)):
            lines, st = convert_labels(folder / f"{stem}.xml")
            dst = out_root / f"labels_{label_set}" / split / f"{stem}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
            for k, v in st.items():
                total[f"{label_set}_{k}"] += v
    total["images"] = len(stems)
    return total


def activate_label_set(out_root: Path, modality: str, label_set: str) -> int:
    n = 0
    for split in SPLITS:
        src = out_root / f"labels_{label_set}" / split
        dst = out_root / modality / "labels" / split
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            continue
        for f in src.glob("*.txt"):
            shutil.copy2(f, dst / f.name)
            n += 1
    return n
