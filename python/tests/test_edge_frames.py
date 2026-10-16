"""Letterbox ön/son işleme ve modalite yükleme testleri (Görev 5)."""

import numpy as np
import pytest

from baykus.data.multichannel import stack_rgbt, write_rgbt4
from baykus.edge.frames import Letterbox, load_input, postprocess, preprocess
from baykus.io import imwrite_any


def test_preprocess_pads_640x512_and_flips_bgr():
    img = np.zeros((512, 640, 3), np.uint8)
    img[..., 0], img[..., 2] = 10, 30  # B=10, R=30
    x, lb = preprocess(img, 640)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert lb == Letterbox(gain=1.0, pad_x=0, pad_y=64, width=640, height=512)
    assert x[0, 0, 64, 0] == pytest.approx(30 / 255) and x[0, 2, 64, 0] == pytest.approx(10 / 255)
    assert x[0, :, 0, 0] == pytest.approx([114 / 255] * 3)


def test_preprocess_keeps_rgbt_order():
    img = np.dstack([np.full((512, 640), v, np.uint8) for v in (1, 2, 3, 4)])
    x, _ = preprocess(img)
    assert (x[0, :, 64, 0] * 255).round().tolist() == [1, 2, 3, 4]


def test_preprocess_scales_up_small_image():
    _, lb = preprocess(np.zeros((256, 320, 3), np.uint8))
    assert lb == Letterbox(gain=2.0, pad_x=0, pad_y=64, width=320, height=256)


def test_postprocess_maps_back_clips_filters_and_sorts():
    lb = Letterbox(1.0, 0, 64, 640, 512)
    out = np.array(
        [
            [
                [0, 60, 700, 70, 0.5, 0],
                [10, 74, 30, 94, 0.9, 2],
                [5, 70, 6, 71, 0.0005, 1],
                [5, 70, 5, 80, 0.8, 1],
            ]
        ],
        np.float32,
    )
    dets = postprocess(out, lb, conf=0.001, image_id=7)
    assert dets == [
        {
            "image_id": 7,
            "category_id": 3,
            "bbox": pytest.approx([10, 10, 20, 20]),
            "score": pytest.approx(0.9),
        },
        {
            "image_id": 7,
            "category_id": 1,
            "bbox": pytest.approx([0, 0, 640, 6]),
            "score": pytest.approx(0.5),
        },
    ]  # düşük skor ve sıfır genişlik atılır


def test_postprocess_empty():
    lb = Letterbox(1.0, 0, 64, 640, 512)
    assert postprocess(np.zeros((1, 300, 6), np.float32), lb, 0.25, 1) == []


def test_load_input_modalities(tmp_path):
    root = tmp_path / "edge"
    bgr = np.full((32, 40, 3), (10, 60, 200), np.uint8)
    ir = np.full((32, 40), 120, np.uint8)
    imwrite_any(root / "rgb/images/val/00001.jpg", bgr)
    imwrite_any(root / "ir/images/val/00001.png", ir)
    rgbt = stack_rgbt(bgr, ir)
    write_rgbt4(root / "rgbt4/images/val/00001.tif", rgbt)

    rgb = load_input(root, "val", "00001", "rgb")
    assert rgb.shape == (32, 40, 3) and rgb.dtype == np.uint8
    assert np.abs(rgb.astype(int) - bgr.astype(int)).max() <= 3  # JPEG kaybı

    ir3 = load_input(root, "val", "00001", "ir")
    assert ir3.shape == (32, 40, 3)
    assert np.array_equal(ir3[..., 0], ir3[..., 1])
    assert np.array_equal(ir3[..., 1], ir3[..., 2])
    assert np.all(ir3 == 120)

    assert np.array_equal(load_input(root, "val", "00001", "rgbt4"), rgbt)


def test_load_input_unknown_modality(tmp_path):
    with pytest.raises(ValueError):
        load_input(tmp_path, "val", "00001", "x")
