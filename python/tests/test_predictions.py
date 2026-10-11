from pathlib import Path

from pycocotools.coco import COCO

from baykus.eval.predictions import load_predictions, remap_to_gt, save_predictions


def _gt(stems: list[str]) -> COCO:
    gt = COCO()
    gt.dataset = {
        "images": [
            {"id": i, "file_name": s, "width": 640, "height": 512} for i, s in enumerate(stems, 1)
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "car"}],
    }
    gt.createIndex()
    return gt


def _p(image_id, file_name=None):
    d = {"image_id": image_id, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.5}
    if file_name:
        d["file_name"] = file_name
    return d


def test_excluded_stem_shifts_ids_and_mapping_uses_file_name():
    # 00002 bütünlük kontrolünde dışlandı → 00003'ün bizim kimliğimiz 2 (Review Focus 1)
    gt = _gt(["00001", "00003", "00004"])
    out, dropped = remap_to_gt([_p(3, "00003.jpg"), _p(4, "00004.tif")], gt)
    assert [d["image_id"] for d in out] == [2, 3] and dropped == 0


def test_numeric_image_id_fallback_without_file_name():
    gt = _gt(["00001", "00003"])
    out, dropped = remap_to_gt([_p(3)], gt)
    assert out[0]["image_id"] == 2 and dropped == 0


def test_unknown_images_dropped_and_counted():
    gt = _gt(["00001"])
    out, dropped = remap_to_gt([_p(9, "00009.jpg"), _p(1, "00001.jpg")], gt)
    assert len(out) == 1 and dropped == 1


def test_output_types_are_plain():
    out, _ = remap_to_gt([_p(1, "00001.jpg")], _gt(["00001"]))
    d = out[0]
    assert set(d) == {"image_id", "category_id", "bbox", "score"}
    assert isinstance(d["score"], float) and all(isinstance(x, float) for x in d["bbox"])


def test_gzip_roundtrip(tmp_path: Path):
    preds = [{"image_id": 1, "category_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.5}]
    p = save_predictions(tmp_path / "x" / "p.json.gz", preds)
    assert p.exists() and load_predictions(p) == preds
