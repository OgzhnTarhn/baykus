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


def _run(
    tmp_path: Path, monkeypatch, write_preds: bool = True, n_boot: int = 10, **kw
) -> tuple[list[Path], type]:
    fake = install(monkeypatch)
    fake.write_preds = write_preds
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
        n_boot=n_boot,
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


def test_stale_ultralytics_predictions_are_not_reused(tmp_path: Path, monkeypatch):
    # Önceki koşunun dosyası kalmış; bu koşuda model hiç tahmin vermiyor → 0 tahmin, mAP 0
    stale = tmp_path / "eval" / f"{RUN}_val" / "predictions.json"
    stale.parent.mkdir(parents=True)
    old = [{"image_id": 101, "category_id": 1, "bbox": [16, 20, 32, 20], "score": 0.9}]
    stale.write_text(json.dumps(old), encoding="utf-8")
    paths, _ = _run(tmp_path, monkeypatch, write_preds=False)
    ir = json.loads(paths[0].read_text(encoding="utf-8"))
    assert ir["metrics"]["n_dets"] == 0 and ir["metrics"]["map50"] == 0.0


def _single_class_gt(root: Path) -> None:
    """M3OT gibi: GT tek kategori 'vehicle' (id 1)."""
    for ls in ("ir", "rgb"):
        p = root / "coco" / f"val_{ls}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        d["categories"] = [{"id": 1, "name": "vehicle"}]
        for a in d["annotations"]:
            a["category_id"] = 1
        p.write_text(json.dumps(d), encoding="utf-8")


def test_class_agnostic_m3ot_naming_overrides_and_categories(tmp_path: Path, monkeypatch):
    fake = install(monkeypatch)
    fake.force_category = 3  # model 'bus' diyor; sınıfsız değerlendirmede yine doğru sayılmalı
    root = make_synthetic_dataset(tmp_path / "m3ot")
    _single_class_gt(root)
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
        dataset="m3ot",
        class_agnostic=True,
        results_subdir="m3ot",
    )
    assert [p.name for p in paths] == [f"{RUN}_m3ot_val_ir.json", f"{RUN}_m3ot_val_rgb.json"]
    assert all(p.parent == tmp_path / "results" / "m3ot" for p in paths)
    rec = json.loads(paths[0].read_text(encoding="utf-8"))
    assert rec["dataset"] == "m3ot" and rec["class_agnostic"] is True
    assert rec["metrics"]["map50"] == pytest.approx(1.0)
    call = fake.calls[0]
    assert call["single_cls"] is True and call["agnostic_nms"] is True
    assert call["name"] == f"{RUN}_m3ot_val"
    from baykus.eval.predictions import load_predictions

    dets = load_predictions(tmp_path / "eval" / f"{RUN}_m3ot_val" / "predictions_coco.json.gz")
    assert dets and {d["category_id"] for d in dets} == {1}


def test_default_dataset_keeps_h3_names_and_no_overrides(tmp_path: Path, monkeypatch):
    paths, fake = _run(tmp_path, monkeypatch)
    rec = json.loads(paths[0].read_text(encoding="utf-8"))
    assert rec["dataset"] == "dronevehicle" and rec["class_agnostic"] is False
    assert "single_cls" not in fake.calls[0] and "agnostic_nms" not in fake.calls[0]


def test_run_name():
    from baykus.eval.cli import run_name

    assert run_name("r", "val") == "r_val"
    assert run_name("r", "test", "m3ot") == "r_m3ot_test"
    assert run_name("r", "val", condition={"id": "drop_rgb"}) == "r_val_drop_rgb"


def test_zero_boot_records_null_intervals(tmp_path: Path, monkeypatch):
    paths, _ = _run(tmp_path, monkeypatch, n_boot=0, workers=2)
    rec = json.loads(paths[0].read_text(encoding="utf-8"))
    assert rec["n_boot"] == 0 and rec["metrics"]["ci95"]["map50"] == [None, None]
    assert rec["metrics"]["ci95"]["per_illumination"]["day"] == [None, None]
    assert isinstance(rec["metrics"]["map50"], float)


def test_run_name_variant():
    from baykus.eval.cli import run_name

    assert run_name("r", "val", variant="onnx_fp16") == "r_val_onnx_fp16"
    assert run_name("r", "test", "m3ot", variant="v") == "r_m3ot_test_v"
    assert run_name("r", "val", condition={"id": "drop_rgb"}, variant="v") == "r_val_v_drop_rgb"


def test_eval_record_format_default_and_override():
    from baykus.eval.cli import eval_record

    base = dict(
        run_id="r",
        config="c",
        config_hash="h",
        git={"git_commit": "x", "git_dirty": False},
        seed=42,
        dataset="dronevehicle",
        split="val",
        label_set="ir",
        model={},
        weights=None,
        n_boot=0,
        dropped=0,
        metrics={},
    )
    rec = eval_record(**base)
    assert rec["format"] == "pytorch" and rec["precision"] == "fp32"
    rec = eval_record(**base, fmt="onnx", precision="int8")
    assert rec["format"] == "onnx" and rec["precision"] == "int8"
