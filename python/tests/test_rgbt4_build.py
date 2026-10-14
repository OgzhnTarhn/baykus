from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from baykus.data.cli import main as data_main
from baykus.data.multichannel import build_rgbt4
from baykus.io import imread_any, imwrite_any


def _make_pairs(root: Path, split: str, stems: list[str]) -> None:
    rng = np.random.default_rng(len(split))
    for s in stems:
        rgb = rng.integers(0, 255, (16, 20, 3), dtype=np.uint8)
        imwrite_any(root / "rgb/images" / split / f"{s}.jpg", rgb)
        ir = rng.integers(0, 255, (16, 20), dtype=np.uint8)
        imwrite_any(root / "ir/images" / split / f"{s}.png", ir)


def _pages(p: Path) -> np.ndarray:
    with tifffile.TiffFile(p) as t:
        return np.stack([pg.asarray() for pg in t.pages], axis=-1)  # (H, W, 4)


def test_build_rgbt4_channel_order(tmp_path: Path):
    _make_pairs(tmp_path, "val", ["00001"])
    assert build_rgbt4(tmp_path, splits=["val"]) == 1
    got = _pages(tmp_path / "rgbt4/images/val/00001.tif")
    bgr = imread_any(tmp_path / "rgb/images/val/00001.jpg", cv2.IMREAD_COLOR)
    ir = imread_any(tmp_path / "ir/images/val/00001.png", cv2.IMREAD_GRAYSCALE)
    assert got.shape == (16, 20, 4)
    np.testing.assert_array_equal(got[..., 0], bgr[..., 2])  # R
    np.testing.assert_array_equal(got[..., 2], bgr[..., 0])  # B
    np.testing.assert_array_equal(got[..., 3], ir)  # T


def test_build_rgbt4_only_requested_splits(tmp_path: Path):
    _make_pairs(tmp_path, "val", ["00001"])
    _make_pairs(tmp_path, "test", ["00002"])
    assert build_rgbt4(tmp_path, splits=["val"]) == 1
    assert not (tmp_path / "rgbt4/images/test").exists()


def test_build_rgbt4_parallel_matches_serial(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    stems = ["00001", "00002", "00003"]
    for r in (a, b):
        _make_pairs(r, "train", stems)
    assert build_rgbt4(a, splits=["train"], workers=1) == 3
    assert build_rgbt4(b, splits=["train"], workers=2) == 3
    for s in stems:
        np.testing.assert_array_equal(
            _pages(a / f"rgbt4/images/train/{s}.tif"), _pages(b / f"rgbt4/images/train/{s}.tif")
        )


def test_build_rgbt4_missing_ir_names_the_file(tmp_path: Path):
    _make_pairs(tmp_path, "val", ["00001"])
    (tmp_path / "ir/images/val/00001.png").unlink()
    with pytest.raises(OSError, match="00001.png"):
        build_rgbt4(tmp_path, splits=["val"])


def test_cli_rgbt4_splits_and_workers(tmp_path: Path):
    _make_pairs(tmp_path, "val", ["00001", "00002"])
    _make_pairs(tmp_path, "test", ["00003"])
    assert data_main(["rgbt4", "--out", str(tmp_path), "--splits", "val", "--workers", "2"]) == 0
    assert len(list((tmp_path / "rgbt4/images/val").glob("*.tif"))) == 2
    assert not (tmp_path / "rgbt4/images/test").exists()
    assert (tmp_path / "data_rgbt4.yaml").exists()
