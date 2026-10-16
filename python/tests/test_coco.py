import json

from pycocotools.coco import COCO

from baykus.data.coco import illumination_slices, yolo_to_coco


def _write(d, stem, text):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.txt").write_text(text, encoding="utf-8")


def test_yolo_to_coco_basic(tmp_path):
    d = tmp_path / "labels"
    _write(d, "00002", "0 0.5 0.5 0.1 0.1\n4 0.25 0.25 0.05 0.1\n")
    _write(d, "00001", "")  # nesnesiz görüntü
    coco = yolo_to_coco(d, ["00002", "00001"], 640, 512)
    assert [im["file_name"] for im in coco["images"]] == ["00001", "00002"]
    assert [im["id"] for im in coco["images"]] == [1, 2]
    a0 = coco["annotations"][0]
    assert a0["image_id"] == 2 and a0["category_id"] == 1
    assert a0["bbox"] == [288.0, 230.4, 64.0, 51.2]
    assert a0["area"] == 64.0 * 51.2
    assert coco["annotations"][1]["category_id"] == 5
    assert [c["name"] for c in coco["categories"]] == ["car", "truck", "bus", "van", "freight_car"]


def test_pycocotools_can_load(tmp_path):
    d = tmp_path / "labels"
    _write(d, "a", "1 0.5 0.5 0.2 0.2\n")
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(yolo_to_coco(d, ["a"], 640, 512)), encoding="utf-8")
    gt = COCO(str(p))
    assert len(gt.getAnnIds()) == 1


def test_illumination_slices(tmp_path):
    csv = tmp_path / "illumination.csv"
    csv.write_text(
        "split,stem,brightness,illumination\ntrain,a,10,dark_night\nval,b,200,day\nval,c,90,night\n",
        encoding="utf-8",
    )
    s = illumination_slices(csv, "val", {"b": 1, "c": 2})
    assert s == {"day": [1], "night": [2], "dark_night": []}


def test_cli_coco_writes_gt_and_slices(raw_root, tmp_path):
    from baykus.data.cli import main
    from tests.conftest import make_raw_dataset

    for split in ("train", "val", "test"):
        make_raw_dataset(raw_root, split, ["00001", "00002"], brightness={"00001": 10})
    meta, out, fig = tmp_path / "meta", tmp_path / "out", tmp_path / "fig"
    main(["verify", "--raw", str(raw_root), "--meta", str(meta)])
    main(["convert", "--raw", str(raw_root), "--meta", str(meta), "--out", str(out)])
    main(["illum", "--out", str(out), "--meta", str(meta), "--fig", str(fig)])
    assert main(["coco", "--out", str(out), "--meta", str(meta)]) == 0

    for split in ("train", "val", "test"):
        for label_set in ("ir", "rgb"):
            gt = json.loads((out / "coco" / f"{split}_{label_set}.json").read_text("utf-8"))
            assert [im["file_name"] for im in gt["images"]] == ["00001", "00002"]
            assert len(gt["annotations"]) == 2
        sl = json.loads((meta / f"slices_{split}.json").read_text("utf-8"))["illumination"]
        assert sorted(sum(sl.values(), [])) == [1, 2]


def test_yolo_to_coco_single_class(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n3 0.25 0.25 0.1 0.1\n", encoding="utf-8")
    d = yolo_to_coco(tmp_path, ["a"], 40, 32, single_class="vehicle")
    assert d["categories"] == [{"id": 1, "name": "vehicle"}]
    assert {a["category_id"] for a in d["annotations"]} == {1} and len(d["annotations"]) == 2
