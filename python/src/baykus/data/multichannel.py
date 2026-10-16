"""RGB + termal → 4 kanallı TIFF (FR-D5). Biçim ADR 0004'te."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile

from baykus.data.layout import SPLITS
from baykus.io import imread_any

TIFF_LAYOUT = "pages"  # ADR 0004: her kanal ayrı gri sayfa (Spike A)


def stack_rgbt(bgr: np.ndarray, ir: np.ndarray) -> np.ndarray:
    if bgr.shape[:2] != ir.shape[:2]:
        raise ValueError(f"boyut uyuşmazlığı: {bgr.shape[:2]} vs {ir.shape[:2]}")
    ir2 = ir if ir.ndim == 2 else ir[..., 0]
    return np.dstack([bgr[..., 2], bgr[..., 1], bgr[..., 0], ir2]).astype(np.uint8)


def write_rgbt4(path: Path, rgbt: np.ndarray) -> None:
    """Her kanal ayrı gri sayfa. Tek sayfalı RGB+alfa TIFF'te OpenCV, RGB'yi 4. kanalla
    çarpıp BGR'ye çeviriyor (Spike A); çok sayfalıyı Ultralytics sırayla yığıyor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path) as tw:
        for c in range(rgbt.shape[2]):
            tw.write(rgbt[..., c], photometric="minisblack", compression="deflate")


def read_pair(rgb_path: Path, ir_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(BGR, IR gri). Eksik ya da okunamayan dosya adıyla OSError verir."""
    bgr = imread_any(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"okunamadı: {rgb_path}")
    ir = imread_any(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        raise OSError(f"okunamadı: {ir_path}")
    return bgr, ir


def _write_one(job: tuple[Path, Path, Path]) -> None:
    rgb_path, ir_path, out_path = job
    write_rgbt4(out_path, stack_rgbt(*read_pair(rgb_path, ir_path)))


def build_rgbt4(
    root: Path,
    splits: Iterable[str] = SPLITS,
    workers: int = 1,
    stems: Iterable[str] | None = None,
) -> int:
    """rgb/ + ir/ → rgbt4/images/<split>/*.tif. ADR 0004: E3 oturumunda yeniden üretilir.
    stems verilirse yalnız o adlar işlenir (T4'ün 2 çekirdeğinde bütün train'i üretmemek için)."""
    root = Path(root)
    keep = None if stems is None else set(stems)
    jobs = [
        (
            p,
            root / "ir/images" / split / f"{p.stem}.png",
            root / "rgbt4/images" / split / f"{p.stem}.tif",
        )
        for split in splits
        for p in sorted((root / "rgb/images" / split).glob("*.jpg"))
        if keep is None or p.stem in keep
    ]
    if workers <= 1:
        for job in jobs:
            _write_one(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_write_one, jobs, chunksize=32))
    return len(jobs)
