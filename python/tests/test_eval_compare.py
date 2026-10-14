"""baykus-eval compare: iki modelin aynı görüntülerde eşleştirilmiş bootstrap farkı (FR-V8)."""

import json
from pathlib import Path

from baykus.eval.cli import main
from baykus.eval.predictions import save_predictions
from tests.coco_helpers import make_gt, perfect


def _setup(tmp_path: Path, slices: dict) -> list[str]:
    gt = make_gt({i: [(1, [0, 0, 10, 10])] for i in range(1, 21)})
    gt_path = tmp_path / "val_ir.json"
    gt_path.write_text(json.dumps(gt.dataset), encoding="utf-8")
    good = save_predictions(tmp_path / "a.json.gz", perfect(gt))
    half = save_predictions(
        tmp_path / "b.json.gz", [d for d in perfect(gt) if d["image_id"] % 2 == 0]
    )
    sl = tmp_path / "slices_val.json"
    sl.write_text(json.dumps({"illumination": slices}), encoding="utf-8")
    return ["compare", "--gt", str(gt_path), "--a", str(good), "--b", str(half),
            "--slices", str(sl), "--name", "e1_vs_e2", "--results", str(tmp_path / "res"),
            "--n-boot", "100"]  # fmt: skip


def _out(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "res" / "compare" / "e1_vs_e2.json").read_text(encoding="utf-8"))


def test_compare_writes_paired_difference(tmp_path: Path):
    args = _setup(tmp_path, {"day": list(range(1, 11)), "night": list(range(11, 21))})
    assert main(args) == 0
    out = _out(tmp_path)
    assert out["name"] == "e1_vs_e2" and out["n_boot"] == 100 and out["created_at"]
    assert out["overall"]["diff"] > 0 and out["overall"]["significant"] is True
    assert set(out["per_illumination"]) == {"day", "night"}
    assert out["per_illumination"]["day"]["diff"] > 0


def test_compare_empty_slice_is_null(tmp_path: Path):
    # Review Focus 4 ile aynı ilke: görüntüsü olmayan dilim 0 değil null
    assert main(_setup(tmp_path, {"day": [1, 2, 3], "dark_night": []})) == 0
    assert _out(tmp_path)["per_illumination"]["dark_night"] is None


def test_compare_slice_without_gt_images_is_null(tmp_path: Path):
    # Dilim listesi başka bir bölümün kimliklerini taşıyorsa (GT'de yok) çökmemeli
    assert main(_setup(tmp_path, {"day": [1, 2], "night": [999, 1000]})) == 0
    assert _out(tmp_path)["per_illumination"]["night"] is None
