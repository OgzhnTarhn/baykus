from pathlib import Path

import yaml

from baykus.data.cli import main
from baykus.data.yamls import write_data_yamls


def test_write_data_yamls(tmp_path: Path):
    paths = write_data_yamls(tmp_path)
    assert sorted(p.name for p in paths) == ["data_ir.yaml", "data_rgb.yaml", "data_rgbt4.yaml"]
    d = yaml.safe_load((tmp_path / "data_rgbt4.yaml").read_text(encoding="utf-8"))
    assert d["channels"] == 4 and d["path"] == (tmp_path / "rgbt4").as_posix()
    assert d["train"] == "images/train" and d["names"][4] == "freight_car"
    assert yaml.safe_load((tmp_path / "data_ir.yaml").read_text(encoding="utf-8"))["channels"] == 3


def test_yaml_subcommand(tmp_path: Path):
    assert main(["yaml", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "data_rgb.yaml").exists()
