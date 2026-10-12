"""Yarıda kalan eğitimi sürdürme: kayıt baştan yazılır, devamda etiketler yeniden kurulur."""

import datetime as dt
import json
from pathlib import Path

import pytest

from baykus.train.cli import RECORD, ResumeError, finish_record, make_record, prepare_resume
from baykus.train.config import parse_config
from tests.synthetic import make_synthetic_dataset

CFG = parse_config(
    {
        "experiment": "e1_rgb",
        "mode": "rgb",
        "model": "yolo26s.pt",
        "data": "data/dv_rgbt/data_rgb.yaml",
        "train": {"epochs": 50},
    }
)
START = dt.datetime(2026, 10, 12, 21, 0, 0, tzinfo=dt.UTC)
GIT = {"git_commit": "49da0cbdbdb396fe2fc4d573a40cce7cd86152f5", "git_dirty": False}


def _record(run_dir: Path) -> dict:
    return make_record(
        CFG,
        config="../configs/e1_rgb.yaml",
        run_id=run_dir.name,
        git=GIT,
        mlflow_run_id="m1",
        smoke=False,
        epochs=50,
        run_dir=run_dir,
        start=START,
    )


def _run_dir(tmp_path: Path, with_last: bool = True) -> Path:
    run_dir = tmp_path / "runs" / "2026-10-12_e1-rgb_s_49da0cb"
    (run_dir / "weights").mkdir(parents=True)
    if with_last:
        (run_dir / "weights" / "last.pt").write_bytes(b"x")
    return run_dir


def test_record_starts_as_running(tmp_path: Path):
    r = _record(tmp_path / "r1")
    assert r["status"] == "running" and r["train_seconds"] is None
    assert r["run_id"] == "r1" and r["epochs_requested"] == 50 and r["mlflow_run_id"] == "m1"
    assert r["weights"].endswith("best.pt") and r["created_at"] == "2026-10-12T21:00:00Z"
    assert set(r["hardware"]) == {"device", "driver", "cpu_count"}
    assert r["software"]["python"]


def test_finish_record_accumulates_seconds():
    r = finish_record({"train_seconds": None}, 100.04, START)
    assert r["train_seconds"] == 100.0 and r["status"] == "finished"
    assert r["finished_at"] == "2026-10-12T21:00:00Z"
    assert finish_record({"train_seconds": 100.0}, 50.0, START)["train_seconds"] == 150.0


def test_resume_needs_last_checkpoint(tmp_path: Path):
    root = make_synthetic_dataset(tmp_path / "dv")
    with pytest.raises(ResumeError, match="last.pt"):
        prepare_resume(CFG, _run_dir(tmp_path, with_last=False), root, "c.yaml")


def test_resume_needs_data(tmp_path: Path):
    with pytest.raises(ResumeError, match="data_rgb.yaml"):
        prepare_resume(CFG, _run_dir(tmp_path), tmp_path / "yok", "c.yaml")


def test_resume_activates_labels_for_new_session(tmp_path: Path):
    # Yeni Colab oturumunda veri yeniden iner; rgb/labels yoktur (Ultralytics etiketsiz eğitirdi)
    root = make_synthetic_dataset(tmp_path / "dv")
    assert not (root / "rgb" / "labels").exists()
    _, data = prepare_resume(CFG, _run_dir(tmp_path), root, "c.yaml")
    assert data == root / "data_rgb.yaml"
    assert len(list((root / "rgb" / "labels" / "train").glob("*.txt"))) == 8


def test_resume_reconstructs_missing_record(tmp_path: Path):
    # Eski sürüm kaydı sadece eğitim sonunda yazıyordu; 38. epoch'ta kopan E1'in kaydı yok
    root = make_synthetic_dataset(tmp_path / "dv")
    run_dir = _run_dir(tmp_path)
    record, _ = prepare_resume(CFG, run_dir, root, "../configs/e1_rgb.yaml")
    assert record["run_id"] == run_dir.name and record["record_reconstructed"] is True
    assert record["git_commit"] == "49da0cb" and record["git_dirty"] is None
    assert record["mlflow_run_id"] is None and record["epochs_requested"] == 50
    assert record["config_hash"] == CFG.config_hash and record["status"] == "running"
    assert json.loads((run_dir / RECORD).read_text(encoding="utf-8"))["run_id"] == run_dir.name


def test_resume_keeps_existing_record(tmp_path: Path):
    root = make_synthetic_dataset(tmp_path / "dv")
    run_dir = _run_dir(tmp_path)
    (run_dir / RECORD).write_text(json.dumps(_record(run_dir)), encoding="utf-8")
    record, _ = prepare_resume(CFG, run_dir, root, "c.yaml")
    assert record["mlflow_run_id"] == "m1" and "record_reconstructed" not in record
