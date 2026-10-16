"""Letterbox ön/son işleme ve modalite yükleme (H5). Ultralytics LetterBox(auto=False)
kuralıyla aynı; 3 kanalda BGR → RGB, 4 kanalda kanal sırası korunur."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tifffile

from baykus.io import imread_any

MODALITIES = ("rgb", "ir", "rgbt4")
PAD_VALUE = 114


@dataclass(frozen=True)
class Letterbox:
    gain: float
    pad_x: int
    pad_y: int
    width: int
    height: int


def preprocess(img: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, Letterbox]:
    h, w = img.shape[:2]
    gain = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * gain), round(h * gain)
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:  # cv2.resize tek kanalı 2 boyuta düşürür
            img = img[..., None]
    dw, dh = (imgsz - new_w) / 2, (imgsz - new_h) / 2
    top, left = round(dh - 0.1), round(dw - 0.1)  # alt ve sağ dolgu kalan boşluk
    channels = img.shape[2] if img.ndim == 3 else 1
    canvas = np.full((imgsz, imgsz, channels), PAD_VALUE, np.uint8)
    canvas[top : top + new_h, left : left + new_w] = img
    if channels == 3:
        canvas = canvas[..., ::-1]
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), Letterbox(gain, left, top, w, h)


def postprocess(out: np.ndarray, lb: Letterbox, conf: float, image_id: int) -> list[dict]:
    dets: list[dict] = []
    for row in out[0]:
        score = float(row[4])
        if score < conf:
            continue
        x1 = (float(row[0]) - lb.pad_x) / lb.gain
        y1 = (float(row[1]) - lb.pad_y) / lb.gain
        x2 = (float(row[2]) - lb.pad_x) / lb.gain
        y2 = (float(row[3]) - lb.pad_y) / lb.gain
        x1, x2 = min(max(x1, 0.0), float(lb.width)), min(max(x2, 0.0), float(lb.width))
        y1, y2 = min(max(y1, 0.0), float(lb.height)), min(max(y2, 0.0), float(lb.height))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        dets.append(
            {
                "image_id": int(image_id),
                "category_id": int(row[5]) + 1,
                "bbox": [x1, y1, w, h],
                "score": score,
            }
        )
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def load_input(
    root: Path, split: str, stem: str, modality: Literal["rgb", "ir", "rgbt4"]
) -> np.ndarray:
    root = Path(root)
    if modality == "rgb":
        path = root / "rgb/images" / split / f"{stem}.jpg"
        img = imread_any(path, cv2.IMREAD_COLOR)
    elif modality == "ir":
        path = root / "ir/images" / split / f"{stem}.png"
        img = imread_any(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif modality == "rgbt4":
        path = root / "rgbt4/images" / split / f"{stem}.tif"
        # Her kanal ayrı sayfa (ADR 0004); Ultralytics gibi sayfalar sırayla yığılır
        try:
            with tifffile.TiffFile(path) as t:
                img = np.stack([p.asarray() for p in t.pages], axis=-1)
        except OSError as exc:
            raise OSError(f"okunamadı: {path}") from exc
    else:
        raise ValueError(f"bilinmeyen modalite: {modality!r}; seçenekler: {MODALITIES}")
    if img is None:
        raise OSError(f"okunamadı: {path}")
    return img
