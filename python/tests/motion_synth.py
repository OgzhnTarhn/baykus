"""Hareket testleri için sentetik video: dokulu arka plan + kayan kamera + kutu biçimli araçlar.

Kamera her karede `cam` px kayar (görüntüde arka plan −cam yönünde kayar); araçların harita konumu
iz başına verilir (durağan iz harita konumunu korur, hareketli iz her karede `speed` px ilerler).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from baykus.io import imwrite_any

W, H = 320, 256


def texture(seed: int = 0, h: int = 600, w: int = 900) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 90, np.uint8)
    for _ in range(900):
        x, y = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
        bw, bh = int(rng.integers(3, 18)), int(rng.integers(3, 18))
        img[y : y + bh, x : x + bw] = int(rng.integers(0, 255))
    return cv2.GaussianBlur(img, (3, 3), 0)


def make_video(
    n: int = 40,
    cam: tuple[float, float] = (1.0, 0.5),
    tracks: dict[int, dict] | None = None,
    size: int = 14,
) -> tuple[dict[int, np.ndarray], dict[int, dict[int, tuple[float, float, float, float]]]]:
    """→ (kare → gri görüntü, iz → kare → kutu). `tracks[id] = {"start": (x, y), "speed": (vx, vy)}`
    harita (ilk kare) koordinatında."""
    tracks = tracks or {}
    bg = texture()
    frames: dict[int, np.ndarray] = {}
    gt: dict[int, dict[int, tuple[float, float, float, float]]] = {t: {} for t in tracks}
    for f in range(1, n + 1):
        ox, oy = 100 + cam[0] * (f - 1), 100 + cam[1] * (f - 1)
        m = np.float32([[1, 0, -ox], [0, 1, -oy]])
        img = cv2.warpAffine(bg, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for tid, t in tracks.items():
            cx = t["start"][0] + t["speed"][0] * (f - 1) - (ox - 100)
            cy = t["start"][1] + t["speed"][1] * (f - 1) - (oy - 100)
            x1, y1 = cx - size / 2, cy - size / 2
            cv2.rectangle(
                img, (round(x1), round(y1)), (round(x1) + size - 1, round(y1) + size - 1), 255, -1
            )
            gt[tid][f] = (x1, y1, x1 + size, y1 + size)
        frames[f] = img
    return frames, gt


class MemFrames:
    """`images.SequenceFrames` arayüzü (name, split, frames, ir, rgb) bellekteki karelerle."""

    def __init__(self, frames: dict[int, np.ndarray], name: str = "syn", split: str = "val"):
        self._f, self.name, self.split, self.frames = frames, name, split, sorted(frames)

    def ir(self, n: int) -> np.ndarray:
        return self._f[n]

    def rgb(self, n: int) -> np.ndarray:
        return cv2.cvtColor(self._f[n], cv2.COLOR_GRAY2BGR)


def write_sequence(root: Path, name: str, frames: dict[int, np.ndarray], gt: dict, split="val"):
    """`data/m3ot_dev` düzeni: sequences/<ad>/{rgb,ir,index.json} ve labels/tracks_ir.csv."""
    d = root / "sequences" / name
    for f, img in frames.items():
        imwrite_any(d / "ir" / f"{f:06d}.png", img)
        imwrite_any(d / "rgb" / f"{f:06d}.jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    idx = {"name": name, "split": split, "frames": sorted(frames), "fps": 30, "size": [W, H]}
    (d / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    lab = root / "labels" / "tracks_ir.csv"
    lab.parent.mkdir(parents=True, exist_ok=True)
    new = not lab.exists()
    with lab.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(
                [
                    "split",
                    "modality",
                    "seq",
                    "frame",
                    "stem",
                    "track_id",
                    "x",
                    "y",
                    "w",
                    "h",
                    "flag",
                    "cls",
                    "vis",
                ]
            )
        for tid, per in gt.items():
            for f, (x1, y1, x2, y2) in per.items():
                w.writerow(
                    [
                        split,
                        "ir",
                        name,
                        f,
                        f"{name}_{f:06d}",
                        tid,
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                        1,
                        1,
                        1.0,
                    ]
                )
    return d
