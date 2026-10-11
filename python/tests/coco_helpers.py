"""COCO test yardımcıları (test_coco_eval, test_bootstrap, test_eval_metrics)."""

from pycocotools.coco import COCO


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
