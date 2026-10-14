"""Gerçek Ultralytics ile: sentetik veride 1 epoch rgb modeli → baykus-robust run (drop_rgb).
Yerelde: uv run pytest -m slow tests/test_e2e_robust.py -v"""

import json
from pathlib import Path

import pytest

from tests.synthetic import SIZE, make_synthetic_dataset

pytestmark = pytest.mark.slow


def test_robust_run_with_real_ultralytics(tmp_path: Path):
    pytest.importorskip("ultralytics")
    from baykus.robustness.cli import main as robust_main
    from baykus.train.cli import main as train_main

    root = make_synthetic_dataset(tmp_path / "dv", with_ir=True)
    cfg = tmp_path / "e1.yaml"
    cfg.write_text(
        "experiment: e1_rgb\nmode: rgb\nmodel: yolo26n.pt\ndata: data/dv_rgbt/data_rgb.yaml\n"
        f"train:\n  imgsz: {SIZE}\n  batch: 4\n  workers: 0\n",
        encoding="utf-8",
    )
    results = tmp_path / "results"
    args = ["--config", str(cfg), "--data-root", str(root), "--device", "cpu"]
    assert train_main([*args, "--project", str(tmp_path / "runs"), "--epochs", "1",
                       "--results", str(results)]) == 0  # fmt: skip
    rec_path = next((tmp_path / "runs").glob("*/baykus_train.json"))
    weights = json.loads(rec_path.read_text(encoding="utf-8"))["weights"]
    rc = robust_main([
        "run", "--model", f"{cfg}={weights}", "--data-root", str(root),
        "--meta", str(root / "meta"), "--work", str(tmp_path / "work"),
        "--project", str(tmp_path / "proj"), "--results", str(results),
        "--only", "drop_rgb", "--n-boot", "10", "--device", "cpu",
    ])  # fmt: skip
    assert rc == 0
    files = sorted((results / "robust").glob("*.json"))
    assert [f.name.split("_val_")[1] for f in files] == ["clean_ir.json", "drop_rgb_ir.json"]
    rec = json.loads(files[1].read_text(encoding="utf-8"))
    assert rec["robustness"]["id"] == "drop_rgb" and rec["dropped_predictions"] == 0
    assert list((tmp_path / "proj").glob("*/predictions_coco.json.gz"))  # K1
