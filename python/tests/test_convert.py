import numpy as np
import pytest

from baykus.data.cli import main
from baykus.data.convert import activate_label_set, convert_labels, to_single_channel
from baykus.io import imread_any
from tests.conftest import make_raw_dataset, write_xml


def test_to_single_channel():
    g = np.full((4, 4, 3), 7, np.uint8)
    assert to_single_channel(g).shape == (4, 4)
    assert to_single_channel(np.zeros((4, 4), np.uint8)).shape == (4, 4)
    c = np.zeros((4, 4, 3), np.uint8)
    c[..., 2] = 255  # kırmızı (BGR)
    assert to_single_channel(c)[0, 0] == 76  # cv2 BGR2GRAY: 0.299*255


def test_convert_labels_crop_and_drop(tmp_path):
    x = tmp_path / "a.xml"
    write_xml(
        x,
        [
            ("car", 200, 150, 240, 170),  # içeride
            ("truck", 80, 90, 130, 140),  # kenarlığa taşıyor → kırpılır
            ("bus", 10, 10, 60, 60),  # tamamen kenarlıkta → düşer
            ("tank", 300, 300, 320, 320),  # bilinmeyen sınıf
        ],
    )
    lines, stats = convert_labels(x)
    assert stats == {"kept": 2, "dropped": 1, "unknown": 1, "skipped": 0}
    c, cx, cy, w, h = lines[0].split()
    assert c == "0"
    assert float(cx) == pytest.approx(120 / 640, abs=1e-6)  # (100+140)/2 / 640; 6 ondalık
    assert float(cy) == pytest.approx(60 / 512, abs=1e-6)
    assert lines[1].startswith("1 ")


def test_convert_split_and_empty_image(raw_root, tmp_path):
    for split in ("train", "val", "test"):
        make_raw_dataset(
            raw_root, split, ["00001", "00002"], objects={"00002": []}
        )  # 00002'de hiç nesne yok
    meta, out = tmp_path / "meta", tmp_path / "out"
    assert main(["verify", "--raw", str(raw_root), "--meta", str(meta)]) == 0
    assert main(["convert", "--raw", str(raw_root), "--meta", str(meta), "--out", str(out)]) == 0
    rgb = imread_any(out / "rgb/images/train/00001.jpg")
    ir = imread_any(out / "ir/images/train/00001.png")
    assert rgb.shape == (512, 640, 3)
    assert ir.shape == (512, 640)
    assert (out / "labels_ir/train/00002.txt").read_text(encoding="utf-8") == ""
    assert (out / "labels_rgb/val/00001.txt").read_text(encoding="utf-8").startswith("0 ")


def test_activate_label_set(raw_root, tmp_path):
    make_raw_dataset(raw_root, "train", ["00001"])
    for split in ("val", "test"):
        make_raw_dataset(raw_root, split, [])
    meta, out = tmp_path / "meta", tmp_path / "out"
    main(["verify", "--raw", str(raw_root), "--meta", str(meta)])
    main(["convert", "--raw", str(raw_root), "--meta", str(meta), "--out", str(out)])
    n = activate_label_set(out, "rgb", "ir")
    assert n == 1
    assert (out / "rgb/labels/train/00001.txt").exists()
