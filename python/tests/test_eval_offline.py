import json
from pathlib import Path

import pytest

from baykus.eval.cli import evaluate_weights, train_record
from baykus.train.config import load_config
from tests.fake_ultralytics import install, make_weights
from tests.synthetic import make_synthetic_dataset

RUN = "2026-10-20_e1-rgb_s_abc1234"


def _cfg(tmp_path: Path) -> Path:
    p = tmp_path / "e1.yaml"
    p.write_text(
        "experiment: e1_rgb\nmode: rgb\nmodel: yolo26n.pt\ndata: data/dv_rgbt/data_rgb.yaml\n"
        "eval:\n  n_boot: 10\n",
        encoding="utf-8",
    )
    return p


def _run(tmp_path: Path, monkeypatch, **kw) -> tuple[list[Path], type]:
    fake = install(monkeypatch)
    root = make_synthetic_dataset(tmp_path / "dv")
    cfg_path = _cfg(tmp_path)
    paths = evaluate_weights(
        load_config(cfg_path),
        config=str(cfg_path),
        weights=make_weights(tmp_path / "runs" / RUN, RUN),
        data=root / "data_rgb.yaml",
        split="val",
        meta=root / "meta",
        project=tmp_path / "eval",
        results=tmp_path / "results",
        n_boot=10,
        **kw,
    )
    return paths, fake


def test_train_record_missing_is_empty(tmp_path: Path):
    assert train_record(tmp_path / "x" / "weights" / "best.pt") == {}


def test_evaluate_weights_main_path(tmp_path: Path, monkeypatch):
    paths, fake = _run(tmp_path, monkeypatch)
    assert [p.name for p in paths] == [f"{RUN}_val_ir.json", f"{RUN}_val_rgb.json"]
    assert all(p.parent == tmp_path / "results" / "runs" for p in paths)
    ir = json.loads(paths[0].read_text(encoding="utf-8"))
    assert ir["train_run_id"] == RUN and ir["dropped_predictions"] == 0
    assert ir["metrics"]["map50"] == pytest.approx(1.0)
    assert ir["crosscheck"]["abs_diff"] == pytest.approx(0.0)
    assert "robustness" not in ir
    call = fake.calls[0]
    assert call["conf"] == 0.001 and call["max_det"] >= 1000 and call["name"] == f"{RUN}_val"
    assert (tmp_path / "eval" / f"{RUN}_val" / "predictions_coco.json.gz").exists()


def test_evaluate_weights_condition_goes_to_robust(tmp_path: Path, monkeypatch):
    cond = {"id": "darken-0.5", "corruption": "darken", "severity": 0.5, "target": "rgb"}
    paths, fake = _run(
        tmp_path, monkeypatch, label_sets=("ir",), condition=cond, results_subdir="robust"
    )
    assert [p.name for p in paths] == [f"{RUN}_val_darken-0.5_ir.json"]
    assert paths[0].parent == tmp_path / "results" / "robust"
    rec = json.loads(paths[0].read_text(encoding="utf-8"))
    assert rec["robustness"] == cond and rec["label_set"] == "ir"
    assert fake.calls[0]["name"] == f"{RUN}_val_darken-0.5"
