import cv2
import numpy as np
import pytest

from baykus.data.multichannel import stack_rgbt, write_rgbt4


def test_stack_rgbt_order():
    bgr = np.zeros((2, 3, 3), np.uint8)
    bgr[..., 0], bgr[..., 1], bgr[..., 2] = 10, 20, 30  # B, G, R
    ir = np.full((2, 3), 99, np.uint8)
    out = stack_rgbt(bgr, ir)
    assert out.shape == (2, 3, 4)
    assert list(out[0, 0]) == [30, 20, 10, 99]


def test_stack_rgbt_rejects_size_mismatch():
    with pytest.raises(ValueError):
        stack_rgbt(np.zeros((2, 3, 3), np.uint8), np.zeros((3, 3), np.uint8))


def test_roundtrip_through_ultralytics_reader(tmp_path):
    ul = pytest.importorskip("ultralytics.utils.patches")
    rng = np.random.default_rng(0)
    rgbt = rng.integers(0, 255, (16, 20, 4), dtype=np.uint8)
    p = tmp_path / "x.tif"
    write_rgbt4(p, rgbt)
    back = ul.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert back.shape == (16, 20, 4)
    # Ultralytics'in okuduğu sıra, Spike A'da bulunan sıraya eşit olmalı:
    np.testing.assert_array_equal(back, rgbt)


def test_cli_rgbt4_writes_tiffs_and_yamls(raw_root, tmp_path):
    import yaml

    from baykus.data.cli import main
    from tests.conftest import make_raw_dataset

    make_raw_dataset(raw_root, "train", ["00001"])
    for split in ("val", "test"):
        make_raw_dataset(raw_root, split, [])
    meta, out = tmp_path / "meta", tmp_path / "out"
    main(["verify", "--raw", str(raw_root), "--meta", str(meta)])
    main(["convert", "--raw", str(raw_root), "--meta", str(meta), "--out", str(out)])
    assert main(["rgbt4", "--out", str(out)]) == 0

    tif = out / "rgbt4/images/train/00001.tif"
    assert tif.exists()
    for modality, ch in (("rgb", 3), ("ir", 3), ("rgbt4", 4)):
        cfg = yaml.safe_load((out / f"data_{modality}.yaml").read_text(encoding="utf-8"))
        assert cfg["channels"] == ch
        assert cfg["path"] == (out / modality).as_posix()
        assert cfg["names"] == {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}


def test_build_rgbt4_selected_stems(tmp_path):
    from baykus.data.multichannel import build_rgbt4
    from tests.synthetic import make_synthetic_dataset

    root = make_synthetic_dataset(tmp_path / "d", with_ir=True)
    assert build_rgbt4(root, ["train"], stems=["00002"]) == 1
    assert sorted(p.name for p in (root / "rgbt4/images/train").glob("*.tif")) == ["00002.tif"]
