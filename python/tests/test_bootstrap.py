import numpy as np
import pytest

from baykus.eval.bootstrap import (
    ap50,
    bootstrap_ci,
    collect_matches,
    paired_diff_ci,
    precision_recall,
    subset,
)
from baykus.eval.coco_eval import evaluate_coco
from tests.coco_helpers import make_gt, perfect


def _random_case(seed: int = 0, n_img: int = 12):
    rng = np.random.default_rng(seed)
    boxes, dets = {}, []
    for img in range(1, n_img + 1):
        k = int(rng.integers(0, 6))  # 0 GT'li (arka plan) görüntüler de olsun
        boxes[img] = []
        for _ in range(k):
            x, y = rng.uniform(0, 500, 2)
            w, h = rng.uniform(10, 60, 2)
            cat = int(rng.integers(1, 4))
            boxes[img].append((cat, [x, y, w, h]))
            if rng.random() < 0.8:  # bazı nesneler kaçırılsın
                j = rng.normal(0, 3, 4)
                dets.append(
                    {
                        "image_id": img,
                        "category_id": cat,
                        "bbox": [x + j[0], y + j[1], w + j[2], h + j[3]],
                        "score": float(rng.uniform(0.05, 1)),
                    }
                )
        for _ in range(int(rng.integers(0, 3))):  # yanlış pozitifler
            dets.append(
                {
                    "image_id": img,
                    "category_id": int(rng.integers(1, 4)),
                    "bbox": [*rng.uniform(0, 500, 2), 30.0, 30.0],
                    "score": float(rng.uniform(0.05, 1)),
                }
            )
    return make_gt(boxes), dets


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_identity_sample_matches_pycocotools(seed):
    gt, dets = _random_case(seed)
    assert ap50(collect_matches(gt, dets)) == pytest.approx(
        evaluate_coco(gt, dets)["map50"], abs=1e-12
    )


def test_hand_computed_ap():
    # 2 GT; skorlar .9 TP, .8 FP, .7 TP → rc=[.5,.5,1], monoton pr=[1,2/3,2/3]
    # 101 eşikte: r<=0.5 için 1.0 (51 nokta), r>0.5 için 2/3 (50 nokta)
    gt = make_gt({1: [(1, [0, 0, 10, 10]), (1, [100, 100, 10, 10])]})
    dets = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [300, 300, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 10, 10], "score": 0.7},
    ]
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert ap50(collect_matches(gt, dets)) == pytest.approx(expected)
    assert evaluate_coco(gt, dets)["map50"] == pytest.approx(expected)


def test_no_detections():
    gt = make_gt({1: [(1, [0, 0, 10, 10])], 2: []})
    m = collect_matches(gt, [])
    assert ap50(m) == 0.0 and bootstrap_ci(m, n=20) == (0.0, 0.0)


def test_no_images():
    gt = make_gt({1: [(1, [0, 0, 10, 10])]})
    m = subset(collect_matches(gt, perfect(gt)), [])
    assert m.img_ids == [] and bootstrap_ci(m, n=10) == (None, None)


def test_perfect_predictions_ci_is_one():
    gt = make_gt({i: [(1, [0, 0, 10, 10])] for i in range(1, 6)})
    assert bootstrap_ci(collect_matches(gt, perfect(gt)), n=50) == (1.0, 1.0)


def test_bootstrap_deterministic_and_bounded():
    gt, dets = _random_case(5, n_img=20)
    m = collect_matches(gt, dets)
    a, b = bootstrap_ci(m, n=200, seed=3), bootstrap_ci(m, n=200, seed=3)
    assert a == b and 0.0 <= a[0] <= a[1] <= 1.0


def test_subset_matches_slice_evaluation():
    gt, dets = _random_case(7, n_img=15)
    ids = [2, 3, 5, 8, 13]
    assert ap50(subset(collect_matches(gt, dets), ids)) == pytest.approx(
        evaluate_coco(gt, dets, ids)["map50"], abs=1e-12
    )


def test_paired_same_model_is_zero():
    gt, dets = _random_case(2)
    m = collect_matches(gt, dets)
    r = paired_diff_ci(m, m, n=100)
    assert r["diff"] == 0.0 and r["ci95"] == [0.0, 0.0] and r["significant"] is False


def test_paired_better_model_is_significant():
    gt = make_gt({i: [(1, [0, 0, 10, 10])] for i in range(1, 31)})
    good = collect_matches(gt, perfect(gt))
    half = collect_matches(gt, [d for d in perfect(gt) if d["image_id"] % 2 == 0])
    r = paired_diff_ci(good, half, n=300)
    assert r["diff"] > 0 and r["ci95"][0] > 0 and r["significant"] is True


def test_paired_requires_same_images():
    gt = make_gt({1: [(1, [0, 0, 10, 10])], 2: [(1, [0, 0, 10, 10])]})
    m = collect_matches(gt, perfect(gt))
    with pytest.raises(ValueError, match="aynı görüntü"):
        paired_diff_ci(m, subset(m, [1]))


def test_precision_recall_at_conf():
    gt = make_gt({1: [(1, [0, 0, 10, 10]), (1, [100, 100, 10, 10])]})
    dets = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [300, 300, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 10, 10], "score": 0.1},
    ]
    assert precision_recall(collect_matches(gt, dets), conf=0.25) == (0.5, 0.5)
    assert precision_recall(collect_matches(gt, []), conf=0.25) == (None, 0.0)


def test_paired_mean_diff_single_condition_equals_paired_diff():
    from baykus.eval.bootstrap import paired_diff_ci, paired_mean_diff_ci

    gt = make_gt({i: [(1, [0, 0, 10, 10])] for i in range(1, 21)})
    ma = collect_matches(gt, perfect(gt))
    mb = collect_matches(gt, [d for d in perfect(gt) if d["image_id"] % 2 == 0])
    assert paired_mean_diff_ci([ma], [mb], 200, 3) == paired_diff_ci(ma, mb, 200, 3)


def test_paired_mean_diff_averages_conditions():
    from baykus.eval.bootstrap import paired_diff_ci, paired_mean_diff_ci

    gt = make_gt({i: [(1, [0, 0, 10, 10])] for i in range(1, 21)})
    good = collect_matches(gt, perfect(gt))
    half = collect_matches(gt, [d for d in perfect(gt) if d["image_id"] % 2 == 0])
    single = paired_diff_ci(good, half, 200)
    both = paired_mean_diff_ci([good, good], [good, half], 200)  # 1. koşulda fark 0
    assert both["diff"] == pytest.approx(single["diff"] / 2)
    assert both["ci95"][0] <= both["diff"] <= both["ci95"][1]


def test_paired_mean_diff_rejects_mismatch():
    from baykus.eval.bootstrap import paired_mean_diff_ci

    gt = make_gt({i: [(1, [0, 0, 10, 10])] for i in range(1, 5)})
    other = make_gt({i: [(1, [0, 0, 10, 10])] for i in range(1, 4)})
    m, o = collect_matches(gt, perfect(gt)), collect_matches(other, perfect(other))
    with pytest.raises(ValueError, match="koşul"):
        paired_mean_diff_ci([m], [m, m])
    with pytest.raises(ValueError, match="aynı görüntü"):
        paired_mean_diff_ci([m, o], [m, m])


def _legacy_ci(m, n, seed):
    """H4'teki tek çekirdekli gövde, olduğu gibi: paralel sürüm bit bit aynı olmalı."""
    k = len(m.img_ids)
    rng = np.random.default_rng(seed)
    vals = np.array([ap50(m, rng.integers(0, k, k)) for _ in range(n)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def test_parallel_bootstrap_equals_legacy():
    gt, dets = _random_case(5, n_img=13)  # tek sayıda görüntü: örnek üretim sırası önemli
    m = collect_matches(gt, dets)
    want = _legacy_ci(m, 40, 7)
    assert bootstrap_ci(m, 40, 7) == want
    assert bootstrap_ci(m, 40, 7, workers=3) == want


def test_parallel_paired_equals_serial():
    gt, dets = _random_case(6)
    a = collect_matches(gt, dets)
    b = collect_matches(gt, [d for d in dets if d["score"] > 0.3])
    assert paired_diff_ci(a, b, 30, 3, workers=2) == paired_diff_ci(a, b, 30, 3)


def test_zero_boot_gives_none():
    gt, dets = _random_case(8)
    a = collect_matches(gt, dets)
    b = collect_matches(gt, dets[: len(dets) // 2])
    assert bootstrap_ci(a, 0) == (None, None)
    r = paired_diff_ci(a, b, 0)
    assert r["ci95"] == [None, None] and r["significant"] is None
    assert r["diff"] == pytest.approx(ap50(a) - ap50(b))
