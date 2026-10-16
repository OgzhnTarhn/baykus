from pathlib import Path

import pytest
import yaml

from baykus.edge.calib import calibration_stems, write_calib_data
from tests.synthetic import make_synthetic_dataset


def test_calibration_stems_deterministic_sorted_subset(tmp_path):
    root = make_synthetic_dataset(tmp_path / "d", n_train=8, with_ir=True)
    a = calibration_stems(root, n=5, seed=42)
    assert a == calibration_stems(root, n=5, seed=42)
    assert a == sorted(a)
    assert len(set(a)) == 5
    assert set(a) <= {f"{i:05d}" for i in range(1, 9)}
    assert calibration_stems(root, n=50) == [f"{i:05d}" for i in range(1, 9)]


def test_write_calib_data(tmp_path):
    root = make_synthetic_dataset(tmp_path / "d", n_train=8, with_ir=True)
    y = write_calib_data(root, "rgb", ["00001", "00002", "00003"], tmp_path / "calib")
    assert y.name == "data_rgb_calib.yaml"
    d = yaml.safe_load(y.read_text(encoding="utf-8"))
    assert Path(d["train"]).name == "calib_rgb.txt"
    lines = [ln for ln in Path(d["train"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        assert Path(line).is_absolute()
        assert line.endswith(".jpg")
        assert Path(line).exists()
    assert d["channels"] == 3
    assert len(d["names"]) == 5
    assert d["names"][0] == "car"
    assert d["val"] == "images/val"
    assert Path(d["path"]) == (root / "rgb")
    assert (root / "rgb/labels/train/00001.txt").exists()


@pytest.mark.parametrize("modality,ext", [("rgb", ".jpg"), ("ir", ".png")])
def test_write_calib_data_modalities(tmp_path, modality, ext):
    root = make_synthetic_dataset(tmp_path / "d", n_train=8, with_ir=True)
    y = write_calib_data(root, modality, ["00001", "00002"], tmp_path / "calib")
    d = yaml.safe_load(y.read_text(encoding="utf-8"))
    lines = [ln for ln in Path(d["train"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    for line in lines:
        assert line.endswith(ext)
        assert Path(line).exists()
    assert d["channels"] == 3


def test_cli_calib_writes_list_and_rgbt4_only_for_stems(tmp_path):
    from baykus.edge.cli import main

    root = make_synthetic_dataset(tmp_path / "d", n_train=8, with_ir=True)
    cfg = Path(__file__).resolve().parents[2] / "configs/edge/e9_edge.yaml"
    args = ["calib", "--config", str(cfg), "--data-root", str(root), "--modality", "rgbt4",
            "--out", str(tmp_path / "calib"), "--build-rgbt4"]  # fmt: skip
    assert main(args) == 0
    lines = (tmp_path / "calib/calib_rgbt4.txt").read_text(encoding="utf-8").split()
    assert len(lines) == 8 and all(line.endswith(".tif") for line in lines)  # n=500 > 8
    assert len(list((root / "rgbt4/images/train").glob("*.tif"))) == 8
    assert not (root / "rgbt4/images/val").exists()


def test_cli_error_is_error_line(tmp_path, capsys):
    from baykus.edge.cli import main

    args = ["calib", "--config", str(tmp_path / "yok.yaml"), "--data-root", str(tmp_path),
            "--modality", "rgb", "--out", str(tmp_path / "c")]  # fmt: skip
    assert main(args) == 2 and capsys.readouterr().err.startswith("hata: ")
