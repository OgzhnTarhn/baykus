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
