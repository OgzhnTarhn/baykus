"""Gerçek Ultralytics ile sınıfsız değerlendirme: single_cls + agnostic_nms 5 sınıflı YOLO26'da
hatasız çalışıyor ve bütün tahminler tek kategoride. Yerelde:
uv run pytest -m slow tests/test_e2e_m3ot.py -v"""

import json
from pathlib import Path

import pytest

from tests.synthetic import SIZE, make_synthetic_dataset

pytestmark = pytest.mark.slow


def test_class_agnostic_eval_with_real_ultralytics(tmp_path: Path):
    pytest.importorskip("ultralytics")
    from baykus.eval.cli import main as eval_main
    from baykus.eval.predictions import load_predictions
    from baykus.train.cli import main as train_main
    from tests.test_eval_offline import _single_class_gt

    root = make_synthetic_dataset(tmp_path / "dv")
    cfg = tmp_path / "e1.yaml"
    cfg.write_text(
        "experiment: e1_rgb\nmode: rgb\nmodel: yolo26n.pt\n"
        "data: data/dv_rgbt/data_rgb.yaml\n"
        f"train:\n  imgsz: {SIZE}\n  batch: 4\n  workers: 0\n",
        encoding="utf-8",
    )
    results = tmp_path / "results"
    args = ["--config", str(cfg), "--data-root", str(root), "--device", "cpu"]
    assert (
        train_main(
            [*args, "--project", str(tmp_path / "runs"), "--epochs", "1", "--results", str(results)]
        )
        == 0
    )
    weights = json.loads(
        next((tmp_path / "runs").glob("*/baykus_train.json")).read_text(encoding="utf-8")
    )["weights"]
    _single_class_gt(root)
    rc = eval_main(
        [
            "run",
            *args,
            "--weights",
            weights,
            "--meta",
            str(root / "meta"),
            "--project",
            str(tmp_path / "eval"),
            "--results",
            str(results),
            "--dataset",
            "m3ot",
            "--class-agnostic",
            "--n-boot",
            "10",
        ]
    )
    assert rc == 0
    rec = json.loads(
        next((results / "m3ot").glob("*_m3ot_val_ir.json")).read_text(encoding="utf-8")
    )
    assert rec["class_agnostic"] is True and rec["dropped_predictions"] == 0
    gz = next((tmp_path / "eval").glob("*_m3ot_val/predictions_coco.json.gz"))
    assert {d["category_id"] for d in load_predictions(gz)} <= {1}
