import json
from pathlib import Path

import numpy as np
import pytest

from baykus.eval.bootstrap import collect_matches
from baykus.eval.predictions import save_predictions
from baykus.health.calibrate import (
    apply_isotonic,
    calibrate,
    ece,
    fit_isotonic,
    match_labels,
    reliability,
)
from tests.coco_helpers import make_gt, perfect

GT = make_gt({i: [(1, [10, 10, 20, 20])] for i in range(1, 11)})


def _overconfident() -> list[dict]:
    """Her görüntüde bir doğru, bir yanlış kutu; hepsi 0,95 (aşırı güvenli)."""
    wrong = [{**d, "bbox": [300, 300, 20, 20]} for d in perfect(GT)]
    return [{**d, "score": 0.95} for d in perfect(GT) + wrong]


def test_pav_merges_violators_and_equal_blocks():
    x, y = fit_isotonic(np.array([0.1, 0.2, 0.3, 0.4]), np.array([1, 0, 1, 1]))
    assert x.tolist() == [0.2, 0.4] and y.tolist() == [0.5, 1.0]


def test_pav_ties_form_one_point():
    x, y = fit_isotonic(np.array([0.5, 0.5]), np.array([1, 0]))
    assert x.tolist() == [0.5] and y.tolist() == [0.5]


def test_pav_unsorted_input():
    x, y = fit_isotonic(np.array([0.4, 0.1, 0.3, 0.2]), np.array([1, 1, 1, 0]))
    assert x.tolist() == [0.2, 0.4] and y.tolist() == [0.5, 1.0]


def test_pav_monotone_and_mean_preserving():
    rng = np.random.default_rng(0)
    s = rng.random(500)
    lab = (rng.random(500) < s).astype(int)
    x, y = fit_isotonic(s, lab)
    assert np.all(np.diff(y) > 0) and np.all(np.diff(x) > 0)
    assert apply_isotonic(s, x, y).mean() == pytest.approx(lab.mean())


def test_pav_empty_is_error():
    with pytest.raises(ValueError):
        fit_isotonic(np.array([]), np.array([]))


def test_apply_clamps_outside_range():
    x, y = np.array([0.2, 0.4]), np.array([0.5, 1.0])
    got = apply_isotonic(np.array([0.0, 0.2, 0.3, 0.9]), x, y)
    assert got.tolist() == [0.5, 0.5, 1.0, 1.0]


def test_ece_values():
    assert ece(np.full(10, 0.9), np.ones(10)) == pytest.approx(0.1)
    assert ece(np.array([1.0, 1.0]), np.array([1, 1])) == 0.0  # 1,0 son aralıkta
    assert ece(np.array([]), np.array([])) is None  # hesaplanamayan değer null


def test_reliability_bins():
    bins = reliability(np.array([0.05, 0.95, 1.0]), np.array([0, 1, 0]), bins=10)
    assert len(bins) == 10 and bins[0]["n"] == 1 and bins[9]["n"] == 2
    assert bins[9]["acc"] == 0.5 and bins[9]["conf"] == pytest.approx(0.975)
    assert bins[5] == {"lo": 0.5, "hi": pytest.approx(0.6), "n": 0, "conf": None, "acc": None}


def test_match_labels_counts_equal_collect_matches():
    dets = _overconfident()
    s, lab = match_labels(GT, dets)
    m = collect_matches(GT, dets)
    tp = sum(int(c.tp.sum()) for c in m.classes.values())
    fp = sum(int(c.fp.sum()) for c in m.classes.values())
    assert (int(lab.sum()), int(len(lab) - lab.sum())) == (tp, fp) == (10, 10)
    assert len(s) == len(lab)


def test_calibrate_record_and_k4():
    r = calibrate(GT, _overconfident(), GT, _overconfident())
    assert r["table"]["x"] == [0.95] and r["table"]["y"] == [0.5]
    assert r["test"]["ece_before"] == pytest.approx(0.45) and r["test"]["ece_after"] == 0.0
    assert r["test"]["n_eval"] == 20 and r["val"]["n_dets"] == 20
    assert r["k4"]["limit"] == 0.001 and r["k4"]["passed"] is True
    assert len(r["test"]["bins_before"]) == 15
    json.dumps(r, allow_nan=False)


def test_calibrate_ignores_low_raw_scores_in_ece():
    dets = _overconfident() + [{**perfect(GT)[0], "bbox": [500, 400, 9, 9], "score": 0.1}]
    r = calibrate(GT, dets, GT, dets)
    assert r["test"]["n_eval"] == 20  # ham skoru < 0,25 olan ECE'ye girmez


def test_cli_calibrate(tmp_path: Path):
    from baykus.health.cli import main

    coco = tmp_path / "coco"
    coco.mkdir()
    for split in ("val", "test"):
        (coco / f"{split}_ir.json").write_text(json.dumps(GT.dataset), encoding="utf-8")
    v = save_predictions(tmp_path / "v.json.gz", _overconfident())
    t = save_predictions(tmp_path / "t.json.gz", _overconfident())
    out = tmp_path / "res" / "cal.json"
    args = ["calibrate", "--val-preds", str(v), "--test-preds", str(t)]
    assert main([*args, "--coco-dir", str(coco), "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert len(rec["sources"]["val_sha256"]) == 64 and rec["table"]["y"] == [0.5]
    assert rec["model"] == "e3c_modality_dropout · onnx_fp32"


def test_cli_prints_on_a_cp1254_console(tmp_path: Path, monkeypatch):
    # Windows konsolu (cp1254) ✅, ≤, → yazamaz: komut sonucu yazdıktan sonra çökmemeli
    import io
    import sys

    from baykus.health.cli import main

    coco = tmp_path / "coco"
    coco.mkdir()
    for split in ("val", "test"):
        (coco / f"{split}_ir.json").write_text(json.dumps(GT.dataset), encoding="utf-8")
    v = save_predictions(tmp_path / "v.json.gz", _overconfident())
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1254")
    monkeypatch.setattr(sys, "stdout", console)
    args = ["calibrate", "--val-preds", str(v), "--test-preds", str(v), "--coco-dir", str(coco)]
    assert main([*args, "--out", str(tmp_path / "c.json")]) == 0


def test_match_labels_excludes_detection_on_ignored_gt():
    # crowd (yok sayılan) GT'ye düşen tespit ne doğru ne yanlış: kalibrasyona girmez
    gt = make_gt({1: [(1, [10, 10, 20, 20]), (1, [200, 200, 40, 40])]})
    gt.dataset["annotations"][1]["iscrowd"] = 1
    gt.createIndex()
    dets = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},  # doğru
        {"image_id": 1, "category_id": 1, "bbox": [205, 205, 20, 20], "score": 0.8},  # crowd içi
        {"image_id": 1, "category_id": 1, "bbox": [400, 400, 20, 20], "score": 0.7},  # yanlış
    ]
    s, lab = match_labels(gt, dets)
    assert sorted(zip(s.tolist(), lab.tolist(), strict=True)) == [(0.7, 0), (0.9, 1)]
