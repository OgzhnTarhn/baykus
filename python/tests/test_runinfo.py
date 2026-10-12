import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest

from baykus.runinfo import (
    git_info,
    iso,
    make_run_id,
    model_family,
    model_size,
    software_versions,
    write_json,
)

WHEN = dt.datetime(2026, 10, 20, 14, 0, 0, tzinfo=dt.UTC)


def test_model_name_parsing():
    assert model_family("yolo26s.pt") == "yolo26" and model_size("yolo26s.pt") == "s"
    assert model_size("weights/yolo11n.pt") == "n"
    with pytest.raises(ValueError):
        model_size("best.pt")


def test_run_id_format():
    rid = make_run_id("e3_early_fusion", "yolo26s.pt", "a1b2c3d4e5f6", WHEN)
    assert rid == "2026-10-20_e3-early-fusion_s_a1b2c3d"
    assert make_run_id("e1_rgb", "yolo26n.pt", None, WHEN).endswith("_n_nogit")


def test_iso():
    assert iso(WHEN) == "2026-10-20T14:00:00Z"


def test_git_info_outside_repo(tmp_path: Path):
    assert git_info(tmp_path) == {"git_commit": None, "git_dirty": None}


def test_git_info_in_repo(tmp_path: Path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    git("add", "f.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    info = git_info(tmp_path)
    assert len(info["git_commit"]) == 40 and info["git_dirty"] is False
    (tmp_path / "f.txt").write_text("y", encoding="utf-8")
    assert git_info(tmp_path)["git_dirty"] is True


def test_write_json_utf8(tmp_path: Path):
    p = write_json(tmp_path / "a" / "b.json", {"not": "Karanlık gece"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"not": "Karanlık gece"}
    assert "Karanlık" in p.read_text(encoding="utf-8")


def test_software_versions():
    import platform
    from importlib.metadata import version

    v = software_versions()
    assert set(v) == {"python", "torch", "ultralytics", "numpy", "pycocotools"}
    assert v["python"] == platform.python_version()
    assert v["numpy"] == version("numpy")


def test_hardware_info():
    import os

    from baykus.runinfo import hardware_info

    h = hardware_info()
    assert set(h) == {"device", "driver", "cpu_count"}
    assert h["device"] and h["cpu_count"] == os.cpu_count()


def test_software_versions_missing_package(monkeypatch):
    import baykus.runinfo as ri

    monkeypatch.setattr(ri, "SOFTWARE", ("numpy", "yok-boyle-bir-paket"))
    assert ri.software_versions()["yok-boyle-bir-paket"] is None
