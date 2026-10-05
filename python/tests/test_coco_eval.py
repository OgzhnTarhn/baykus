import pytest
from pycocotools.coco import COCO

from baykus.eval.coco_eval import evaluate_coco, evaluate_slices


def make_gt(boxes_per_image: dict[int, list[tuple[int, list[float]]]]) -> COCO:
    gt = COCO()
    images, anns = [], []
    for img_id, boxes in boxes_per_image.items():
        images.append({"id": img_id, "width": 640, "height": 512})
        for cat, bb in boxes:
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": img_id,
                    "category_id": cat,
                    "bbox": bb,
                    "area": bb[2] * bb[3],
                    "iscrowd": 0,
                }
            )
    gt.dataset = {
        "images": images,
        "annotations": anns,
        "categories": [
            {"id": i, "name": n}
            for i, n in enumerate(["car", "truck", "bus", "van", "freight_car"], 1)
        ],
    }
    gt.createIndex()
    return gt


def perfect(gt: COCO) -> list[dict]:
    return [
        {
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "bbox": a["bbox"],
            "score": 0.9,
        }
        for a in gt.dataset["annotations"]
    ]


def test_perfect_predictions():
    gt = make_gt({1: [(1, [10, 10, 40, 20])], 2: [(2, [100, 100, 60, 60])]})
    r = evaluate_coco(gt, perfect(gt))
    assert r["map50"] == pytest.approx(1.0)
    assert r["map50_95"] == pytest.approx(1.0)
    assert r["per_class"]["car"] == pytest.approx(1.0)
    assert r["per_class"]["bus"] is None  # GT'de olmayan sınıf


def test_empty_predictions_return_zero():
    gt = make_gt({1: [(1, [10, 10, 40, 20])]})
    r = evaluate_coco(gt, [])
    assert r["map50"] == 0.0 and r["n_dets"] == 0


def test_shifted_box_counts_at_50_not_at_95():
    gt = make_gt({1: [(1, [100, 100, 40, 40])]})
    dets = [{"image_id": 1, "category_id": 1, "bbox": [108, 100, 40, 40], "score": 0.9}]
    r = evaluate_coco(gt, dets)  # IoU = 32*40/(2*1600-1280) = 0.667
    assert r["map50"] == pytest.approx(1.0)
    assert r["map50_95"] < 0.5


def test_crowded_image_150_objects_not_capped():
    boxes = [
        (1, [float(10 + (i % 15) * 40), float(10 + (i // 15) * 40), 20.0, 20.0]) for i in range(150)
    ]
    gt = make_gt({1: boxes})
    r = evaluate_coco(gt, perfect(gt))
    assert r["map50"] == pytest.approx(1.0)  # maxDets=100 olsaydı ~0.66 olurdu


def test_slices_restrict_images():
    gt = make_gt({1: [(1, [10, 10, 40, 20])], 2: [(1, [100, 100, 40, 20])]})
    dets = perfect(gt)[:1]  # sadece görüntü 1 doğru tahmin edildi
    r = evaluate_slices(gt, dets, {"day": [1], "night": [2]})
    assert r["day"]["map50"] == pytest.approx(1.0)
    assert r["night"]["map50"] == 0.0
