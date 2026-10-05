import pytest

from baykus.data.report import label_shift, summarize


def _coco(images, anns):
    return {
        "images": [{"id": i, "file_name": f} for i, f in images],
        "annotations": [{"image_id": i, "category_id": c, "bbox": b} for i, c, b in anns],
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}],
    }


def test_label_shift_same_class_within_max_dist():
    ir = _coco([(1, "a")], [(1, 1, [100, 100, 20, 10]), (1, 2, [300, 300, 40, 20])])
    rgb = _coco([(1, "a")], [(1, 1, [103, 104, 20, 10]), (1, 1, [400, 400, 20, 10])])
    r = label_shift(ir, rgb, max_dist=40)
    assert r["matched"] == 1
    assert r["offsets"] == [pytest.approx(5.0)]  # merkez farkı (3, 4)
    assert r["unmatched_ir"] == 1  # truck'ın RGB eşi yok
    assert r["unmatched_rgb"] == 1  # 40 px'ten uzak


def test_label_shift_greedy_prefers_closest():
    ir = _coco([(1, "a")], [(1, 1, [100, 100, 20, 10])])
    rgb = _coco([(1, "a")], [(1, 1, [94, 100, 20, 10]), (1, 1, [102, 100, 20, 10])])
    r = label_shift(ir, rgb)
    assert r["offsets"] == [pytest.approx(2.0)]
    assert r["unmatched_rgb"] == 1


def test_label_shift_joins_images_by_file_name():
    ir = _coco([(1, "a"), (2, "b")], [(1, 1, [0, 0, 10, 10])])
    rgb = _coco([(1, "b"), (2, "a")], [(2, 1, [3, 4, 10, 10])])  # id'ler ters
    r = label_shift(ir, rgb)
    assert r["matched"] == 1 and r["offsets"] == [pytest.approx(5.0)]


def test_label_shift_empty():
    r = label_shift(_coco([(1, "a")], []), _coco([(1, "a")], []))
    assert r == {"matched": 0, "unmatched_ir": 0, "unmatched_rgb": 0, "offsets": []}


def test_summarize():
    coco = {
        "images": [{"id": 1}, {"id": 2}, {"id": 3}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "area": 20 * 20},  # small
            {"image_id": 1, "category_id": 1, "area": 50 * 50},  # medium
            {"image_id": 2, "category_id": 5, "area": 100 * 100},  # large
        ],
        "categories": [
            {"id": i, "name": n}
            for i, n in enumerate(["car", "truck", "bus", "van", "freight_car"], 1)
        ],
    }
    s = summarize(coco)
    assert s["images"] == 3 and s["boxes"] == 3
    assert s["per_class"] == {"car": 2, "truck": 0, "bus": 0, "van": 0, "freight_car": 1}
    assert s["max_per_image"] == 2
    assert s["empty_images"] == 1
    assert s["size"] == {"small": 1, "medium": 1, "large": 1}


def test_cli_report_end_to_end(raw_root, tmp_path):
    from baykus.data.cli import main
    from tests.conftest import make_raw_dataset

    for split in ("train", "val", "test"):
        make_raw_dataset(raw_root, split, ["00001", "00002"], brightness={"00001": 10})
    meta, out, fig = tmp_path / "meta", tmp_path / "out", tmp_path / "fig"
    md = tmp_path / "data-report.md"
    main(["verify", "--raw", str(raw_root), "--meta", str(meta)])
    main(["convert", "--raw", str(raw_root), "--meta", str(meta), "--out", str(out)])
    main(["illum", "--out", str(out), "--meta", str(meta), "--fig", str(fig)])
    main(["coco", "--out", str(out), "--meta", str(meta)])
    args = ["report", "--out", str(out), "--meta", str(meta), "--md", str(md), "--fig", str(fig)]
    assert main(args) == 0

    text = md.read_text(encoding="utf-8")
    assert text.startswith("# E0 — DroneVehicle veri raporu")
    assert "## RGB-IR etiket kayması (train)" in text
    assert (fig / "class_counts_train.png").exists()
    assert (fig / "label_shift_train.png").exists()
