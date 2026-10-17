import pytest

from baykus.eval.metrics import build_metrics
from tests.coco_helpers import make_gt, perfect


def _case():
    gt = make_gt({i: [(1, [0, 0, 10, 10]), (2, [50, 50, 20, 20])] for i in range(1, 7)})
    dets = [d for d in perfect(gt) if not (d["image_id"] == 3 and d["category_id"] == 2)]
    return gt, dets


def test_schema_keys():
    gt, dets = _case()
    m = build_metrics(
        gt, dets, {"day": [1, 2, 3], "night": [4, 5, 6], "dark_night": [1]}, n_boot=50
    )
    assert set(m) >= {
        "map50",
        "map50_95",
        "precision",
        "recall",
        "per_class",
        "per_illumination",
        "per_size",
        "ci95",
        "n_images",
        "n_dets",
    }
    assert set(m["per_class"]) == {"car", "truck", "bus", "van", "freight_car"}
    assert set(m["per_illumination"]) == {"day", "night", "dark_night"}
    lo, hi = m["ci95"]["map50"]
    assert 0.0 <= lo <= hi <= 1.0


def test_empty_slice_is_null_not_zero():
    # Review Focus 4: dilimde hiç görüntü yoksa 0 değil None
    gt, dets = _case()
    m = build_metrics(gt, dets, {"day": [1, 2], "night": [3], "dark_night": []}, n_boot=20)
    assert m["per_illumination"]["dark_night"] is None
    assert m["ci95"]["per_illumination"]["dark_night"] == [None, None]
    assert m["per_illumination"]["day"] == pytest.approx(1.0)


def test_no_illumination_slices():
    gt, dets = _case()
    m = build_metrics(gt, dets, None, n_boot=10)
    assert m["per_illumination"] == {"day": None, "night": None, "dark_night": None}


def test_precision_recall_present():
    gt, dets = _case()
    m = build_metrics(gt, dets, None, n_boot=10)
    assert m["precision"] == pytest.approx(1.0) and m["recall"] == pytest.approx(11 / 12)


def test_slices_reuse_matches_instead_of_reevaluating(monkeypatch):
    """pycocotools bir kez çalışır; dilim mAP50'si eşleştirmelerden (val/test'te süre ≈ yarıya)
    ve değeri dilimi ayrıca değerlendirmekle aynı."""
    import baykus.eval.metrics as bm
    from baykus.eval.coco_eval import evaluate_coco
    from tests.test_bootstrap import _random_case

    gt, dets = _random_case(9, n_img=16)
    slices = {"day": [1, 2, 3, 5, 8], "night": [4, 6, 7, 9, 10, 11], "dark_night": [12, 13, 16]}
    calls = []

    def counting(*a, **kw):
        calls.append(a[2] if len(a) > 2 else kw.get("img_ids"))
        return evaluate_coco(*a, **kw)

    monkeypatch.setattr(bm, "evaluate_coco", counting)
    m = bm.build_metrics(gt, dets, slices, n_boot=10)
    assert calls == [None]  # yalnız tam değerlendirme
    for name, ids in slices.items():
        assert m["per_illumination"][name] == pytest.approx(
            evaluate_coco(gt, dets, ids)["map50"], abs=1e-12
        )
