"""Sentetik minik veriyle baykus-train → baykus-eval. Yerelde: uv run pytest -m slow -v"""

import json
from pathlib import Path

import pytest

from tests.synthetic import SIZE, make_synthetic_dataset

pytestmark = pytest.mark.slow


def test_train_then_eval(tmp_path: Path):
    pytest.importorskip("ultralytics")
    from baykus.eval.cli import main as eval_main
    from baykus.train.cli import main as train_main

    root = make_synthetic_dataset(tmp_path / "dv")
    cfg = tmp_path / "e1.yaml"
    cfg.write_text(
        "experiment: e1_rgb\nmode: rgb\nmodel: yolo26n.pt\ndata: data/dv_rgbt/data_rgb.yaml\n"
        f"train:\n  imgsz: {SIZE}\n  batch: 4\n  workers: 0\n"
        "eval:\n  n_boot: 20\n",
        encoding="utf-8",
    )
    common = [
        "--config",
        str(cfg),
        "--data-root",
        str(root),
        "--device",
        "cpu",
        "--results",
        str(tmp_path / "results"),
    ]
    assert train_main([*common, "--project", str(tmp_path / "runs"), "--epochs", "1"]) == 0
    run_dir = next(p.parent for p in (tmp_path / "runs").glob("*/baykus_train.json"))
    rec = json.loads((run_dir / "baykus_train.json").read_text(encoding="utf-8"))
    assert rec["config_hash"].startswith("sha256:") and Path(rec["weights"]).exists()
    assert rec["software"]["ultralytics"] is not None
    assert (tmp_path / "results" / "train" / f"{rec['run_id']}.json").exists()

    assert (
        eval_main(
            ["run", *common, "--weights", rec["weights"], "--project", str(tmp_path / "eval")]
        )
        == 0
    )
    files = sorted((tmp_path / "results" / "runs").glob("*.json"))
    assert [f.name.rsplit("_", 1)[-1] for f in files] == ["ir.json", "rgb.json"]
    ir = json.loads(files[0].read_text(encoding="utf-8"))
    assert ir["dropped_predictions"] == 0  # val'de 00102 atlandı; eşleme kaymadı
    assert "crosscheck" in ir and ir["metrics"]["per_illumination"]["dark_night"] is None
    assert ir["train_run_id"] == rec["run_id"]
    assert ir["software"]["python"] and ir["software"]["torch"] is not None


class _Interrupt(Exception):
    pass


@pytest.mark.parametrize("old_record_missing", [False, True])
def test_interrupted_training_resumes_in_new_session(
    tmp_path: Path, monkeypatch, old_record_missing: bool
):
    pytest.importorskip("ultralytics")
    import shutil

    from ultralytics.utils.callbacks.base import default_callbacks

    from baykus.train.cli import main as train_main

    def interrupt(trainer):  # 1. epoch kaydedildikten sonra oturum kopmuş gibi
        if trainer.epoch == 0:
            raise _Interrupt

    hooks = [*default_callbacks["on_fit_epoch_end"], interrupt]
    monkeypatch.setitem(default_callbacks, "on_fit_epoch_end", hooks)
    root = make_synthetic_dataset(tmp_path / "dv")
    cfg = tmp_path / "e1.yaml"
    cfg.write_text(
        "experiment: e1_rgb\nmode: rgb\nmodel: yolo26n.pt\ndata: data/dv_rgbt/data_rgb.yaml\n"
        f"train:\n  imgsz: {SIZE}\n  batch: 4\n  workers: 0\n",
        encoding="utf-8",
    )
    common = [
        "--config",
        str(cfg),
        "--data-root",
        str(root),
        "--device",
        "cpu",
        "--results",
        str(tmp_path / "results"),
    ]
    with pytest.raises(_Interrupt):
        train_main([*common, "--project", str(tmp_path / "runs"), "--epochs", "2"])
    run_dir = next((tmp_path / "runs").glob("*_ep2"))
    rec = json.loads((run_dir / "baykus_train.json").read_text(encoding="utf-8"))
    assert rec["status"] == "running" and (run_dir / "weights" / "last.pt").exists()

    monkeypatch.setitem(default_callbacks, "on_fit_epoch_end", hooks[:-1])
    shutil.rmtree(root / "rgb" / "labels")  # yeni Colab oturumu: etiketler kurulmamış
    if old_record_missing:  # eski sürümle kopan E1 (2026-10-12): kayıt hiç yazılmamıştı
        (run_dir / "baykus_train.json").unlink()
    assert train_main([*common, "--resume", str(run_dir)]) == 0
    rec = json.loads((run_dir / "baykus_train.json").read_text(encoding="utf-8"))
    assert rec["status"] == "finished" and rec["resumed_at"]
    assert rec["mlflow_run_id"]  # yeniden kurulan kayıt devamın MLflow kimliğini almalı
    assert (tmp_path / "results" / "train" / f"{rec['run_id']}.json").exists()
    assert len(list((root / "rgb" / "labels" / "train").glob("*.txt"))) == 8
