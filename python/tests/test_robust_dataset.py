from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
import yaml

from baykus.data.multichannel import stack_rgbt
from baykus.io import imread_any
from baykus.robustness.corruptions import CLEAN, Condition, darken
from baykus.robustness.dataset import make_condition
from tests.synthetic import make_synthetic_dataset

ALL = {"rgb", "ir", "rgbt4"}


def _pages(p: Path) -> np.ndarray:
    with tifffile.TiffFile(p) as t:
        return np.stack([pg.asarray() for pg in t.pages], axis=-1)


def _src_pair(src: Path, split: str, stem: str) -> tuple[np.ndarray, np.ndarray]:
    bgr = imread_any(src / "rgb/images" / split / f"{stem}.jpg", cv2.IMREAD_COLOR)
    ir = imread_any(src / "ir/images" / split / f"{stem}.png", cv2.IMREAD_GRAYSCALE)
    return bgr, ir


@pytest.fixture
def src(tmp_path: Path) -> Path:
    return make_synthetic_dataset(tmp_path / "dv", with_ir=True)


def test_clean_condition_is_pixel_identical(src: Path, tmp_path: Path):
    out = tmp_path / "clean"
    assert make_condition(src, out, "val", CLEAN, ALL, seed=42) == 4
    for p in sorted((src / "rgb/images/val").glob("*.jpg")):
        bgr, ir = _src_pair(src, "val", p.stem)
        got_rgb = imread_any(out / "rgb/images/val" / f"{p.stem}.png", cv2.IMREAD_COLOR)
        got_ir = imread_any(out / "ir/images/val" / f"{p.stem}.png", cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(got_rgb, bgr)
        np.testing.assert_array_equal(got_ir, ir)
        np.testing.assert_array_equal(
            _pages(out / "rgbt4/images/val" / f"{p.stem}.tif"), stack_rgbt(bgr, ir)
        )


def test_labels_coco_and_yamls_follow(src: Path, tmp_path: Path):
    out = tmp_path / "c"
    make_condition(src, out, "val", CLEAN, {"rgb"}, seed=42)
    names = sorted(f.name for f in (src / "labels_ir/val").glob("*.txt"))
    assert sorted(f.name for f in (out / "labels_ir/val").glob("*.txt")) == names
    assert (out / "coco/val_ir.json").read_bytes() == (src / "coco/val_ir.json").read_bytes()
    y = yaml.safe_load((out / "data_rgb.yaml").read_text(encoding="utf-8"))
    assert y["path"] == (out / "rgb").as_posix()


def test_only_requested_modalities(src: Path, tmp_path: Path):
    out = tmp_path / "c"
    make_condition(src, out, "val", CLEAN, {"ir"}, seed=42)
    assert (out / "ir/images/val").is_dir()
    assert not (out / "rgb").exists() and not (out / "rgbt4").exists()


def test_darken_applies_to_rgb_only(src: Path, tmp_path: Path):
    out = tmp_path / "d"
    make_condition(src, out, "val", Condition("darken", 0.5, "rgb"), {"rgb", "ir"}, seed=42)
    bgr, ir = _src_pair(src, "val", "00101")
    got = imread_any(out / "rgb/images/val/00101.png", cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(got, darken(bgr, 0.5))
    np.testing.assert_array_equal(
        imread_any(out / "ir/images/val/00101.png", cv2.IMREAD_GRAYSCALE), ir
    )


def test_parallel_noise_matches_serial(src: Path, tmp_path: Path):
    cond = Condition("noise", 10.0, "ir")
    make_condition(src, tmp_path / "a", "val", cond, {"ir"}, seed=42, workers=1)
    make_condition(src, tmp_path / "b", "val", cond, {"ir"}, seed=42, workers=2)
    for p in sorted((tmp_path / "a/ir/images/val").glob("*.png")):
        a = imread_any(p, cv2.IMREAD_GRAYSCALE)
        b = imread_any(tmp_path / "b/ir/images/val" / p.name, cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, _src_pair(src, "val", p.stem)[1])


def test_train_split_supported_for_phase2(src: Path, tmp_path: Path):
    # Kanca K2: bozulmalar train'de de üretilebilir (değerlendirme yine val/test'te)
    n = make_condition(src, tmp_path / "t", "train", Condition("noise", 5.0, "ir"), {"ir"}, 42)
    assert n == 8 and len(list((tmp_path / "t/ir/images/train").glob("*.png"))) == 8


def test_unknown_modality_and_missing_ir(src: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="rgbt"):
        make_condition(src, tmp_path / "x", "val", CLEAN, {"rgbt"}, seed=42)
    (src / "ir/images/val/00101.png").unlink()
    with pytest.raises(OSError, match="00101.png"):
        make_condition(src, tmp_path / "y", "val", CLEAN, {"ir"}, seed=42)


def test_rgb_only_condition_does_not_need_ir(src: Path, tmp_path: Path):
    (src / "ir/images/val/00101.png").unlink()  # RGB modeli için IR okunmamalı
    assert make_condition(src, tmp_path / "r", "val", Condition("darken", 0.5, "rgb"),
                          {"rgb"}, seed=42) == 4  # fmt: skip


def test_parallel_pool_uses_spawn(src: Path, tmp_path: Path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import baykus.robustness.dataset as ds

    seen = {}

    class Spy(ThreadPoolExecutor):  # bağlamı kaydet, işi iş parçacığında yap
        def __init__(self, max_workers=None, mp_context=None):
            seen["ctx"] = mp_context.get_start_method() if mp_context else None
            super().__init__(max_workers)

    monkeypatch.setattr(ds, "ProcessPoolExecutor", Spy)
    make_condition(src, tmp_path / "p", "val", CLEAN, {"ir"}, seed=42, workers=2)
    assert seen["ctx"] == "spawn"


def _zero_ir_hook(bgr: np.ndarray, ir: np.ndarray):  # spawn işçisine gidebilsin: modül düzeyi
    return bgr, np.zeros_like(ir), {"gate": "ir_off"}


@pytest.mark.parametrize("workers", [1, 2])
def test_make_condition_hook_writes_hooked_pixels_and_log(tmp_path: Path, workers: int):
    import json

    src = make_synthetic_dataset(tmp_path / "d", with_ir=True)
    out = tmp_path / "c"
    n = make_condition(src, out, "val", CLEAN, {"rgbt4"}, 42, workers, hook=_zero_ir_hook)
    tifs = sorted((out / "rgbt4/images/val").glob("*.tif"))
    assert len(tifs) == n
    t = _pages(tifs[0])
    assert not t[..., 3].any() and t[..., :3].any()
    lines = (out / "hook.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(x) for x in lines]
    assert len(recs) == n and all(r["gate"] == "ir_off" for r in recs)
    assert [r["stem"] for r in recs] == [p.stem for p in tifs]


def test_make_condition_without_hook_writes_no_log(tmp_path: Path):
    src = make_synthetic_dataset(tmp_path / "d", with_ir=True)
    make_condition(src, tmp_path / "c", "val", CLEAN, {"rgbt4"}, 42)
    assert not (tmp_path / "c/hook.jsonl").exists()
