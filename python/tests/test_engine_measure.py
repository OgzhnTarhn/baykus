import gzip
import json
from pathlib import Path

import pytest

from baykus.engine_ref.cli import main as engine_main
from baykus.engine_ref.measure import (
    DUP_FLOOR,
    ap50_sequence,
    duplicate_rate,
    ece_calibrated,
    ece_raw,
    load_gt,
    measure,
    to_coco,
)
from baykus.track.runner import TRACK_FLOOR

GT_BOXES = {1: [[10, 20, 30, 40]], 2: [[100, 100, 20, 20], [300, 200, 40, 30]]}


def _labels(tmp_path: Path, split: str = "val", seq: str = "1_08") -> Path:
    d = tmp_path / "labels"
    d.mkdir(exist_ok=True)
    images, anns = [], []
    for f, boxes in GT_BOXES.items():
        img_id = 100 + f
        images.append({"id": img_id, "file_name": f"{seq}_{f:06d}", "width": 640, "height": 512})
        for b in boxes:
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": b,
                    "area": b[2] * b[3],
                    "iscrowd": 0,
                }
            )
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "vehicle"}]}
    (d / f"{split}_ir.json").write_text(json.dumps(coco), encoding="utf-8")
    return d


def _det(box_xywh, raw=0.9, cal=0.95, cls_id=0, index=0):
    x, y, w, h = box_xywh
    return {
        "index": index,
        "cls_id": cls_id,
        "score_raw": raw,
        "score": cal,
        "box": [x, y, x + w, y + h],
    }


def _dets(frames: dict[int, list[dict]], split: str = "val", seq: str = "1_08") -> dict:
    return {
        "seq": seq,
        "split": split,
        "frames": [{"frame": f, "dets": d} for f, d in frames.items()],
    }


def _perfect(seq: str = "1_08", split: str = "val") -> dict:
    return _dets({f: [_det(b) for b in boxes] for f, boxes in GT_BOXES.items()}, split, seq)


def test_to_coco_xywh_and_ids(tmp_path: Path):
    gt = load_gt(_labels(tmp_path), "val")
    got = to_coco(_dets({1: [_det([1, 2, 4, 6], raw=0.5, cal=0.6, cls_id=3)]}), gt)
    assert got == [
        {"image_id": 101, "category_id": 1, "bbox": [1, 2, 4, 6], "score": 0.5, "score_cal": 0.6}
    ]


def test_to_coco_unknown_frame_is_error(tmp_path: Path):
    gt = load_gt(_labels(tmp_path), "val")
    with pytest.raises(ValueError, match="1_08_000009"):
        to_coco(_dets({9: []}), gt)


def test_ap50_perfect_is_one(tmp_path: Path):
    gt = load_gt(_labels(tmp_path), "val")
    assert ap50_sequence(to_coco(_perfect(), gt), gt) == pytest.approx(1.0)


def test_ece_none_when_no_dets(tmp_path: Path):
    gt = load_gt(_labels(tmp_path), "val")
    assert ece_raw([], gt) is None and ece_calibrated([], gt) is None


def test_ece_uses_raw_floor_and_calibrated_scores(tmp_path: Path):
    gt = load_gt(_labels(tmp_path), "val")
    dets = _dets({1: [_det([10, 20, 30, 40], raw=0.8, cal=1.0), _det([500, 400, 5, 5], 0.2, 0.0)]})
    coco = to_coco(dets, gt)
    assert ece_raw(coco, gt) == pytest.approx(0.2)  # 0,2 ham skorlu tespit 0,25 altında: dışarıda
    assert ece_calibrated(coco, gt) == pytest.approx(0.0)


def test_duplicate_rate_cross_class():
    frames = {
        1: [
            _det([10, 10, 20, 20], cls_id=0),
            _det([10, 10, 20, 20], cls_id=1),  # aynı nesne, başka sınıf: çift
            _det([200, 200, 20, 20], cls_id=0),
        ],
        2: [_det([10, 10, 20, 20], cls_id=2), _det([10, 10, 20, 20], cls_id=2)],  # aynı sınıf
    }
    assert duplicate_rate(_dets(frames)) == pytest.approx(1 / 5)


def test_duplicate_rate_counts_tracking_input():
    # tasarım §4.3: oran takip girdisinde (ham ≥ 0,1); 0,1-0,25 arası kopyalar da takipçiye gider
    frames = {1: [_det([10, 10, 20, 20], cls_id=0), _det([10, 10, 20, 20], raw=0.1, cls_id=1)]}
    assert duplicate_rate(_dets(frames)) == pytest.approx(1 / 2)


def test_duplicate_rate_ignores_below_track_floor():
    frames = {1: [_det([10, 10, 20, 20], cls_id=0), _det([10, 10, 20, 20], raw=0.09, cls_id=1)]}
    assert duplicate_rate(_dets(frames)) == 0.0


def test_duplicate_floor_is_track_floor():
    assert DUP_FLOOR == TRACK_FLOOR


def test_measure_refuses_test(tmp_path: Path):
    labels = _labels(tmp_path, "test", "1_03")
    with pytest.raises(ValueError, match="i-know-this-is-test"):
        measure([_perfect("1_03", "test")], labels, split="test")
    assert measure([_perfect("1_03", "test")], labels, split="test", allow_test=True)["ap50"] > 0


def test_measure_split_mismatch_is_error(tmp_path: Path):
    with pytest.raises(ValueError, match="split"):
        measure([_perfect(split="train")], _labels(tmp_path), split="val")


def test_measure_reports_counts(tmp_path: Path):
    rec = measure([_perfect()], _labels(tmp_path), split="val")
    assert rec["ap50"] == pytest.approx(1.0) and rec["n_images"] == 2 and rec["n_dets"] == 3
    assert rec["per_seq"]["1_08"]["ap50"] == pytest.approx(1.0)
    assert set(rec) >= {"ece_raw", "ece_calibrated", "duplicate_rate", "sequences"}
    json.dumps(rec, allow_nan=False)


def test_cli_measure(tmp_path: Path):
    f = tmp_path / "1_08.json.gz"
    f.write_bytes(gzip.compress(json.dumps(_perfect()).encode()))
    out = tmp_path / "m.json"
    labels = _labels(tmp_path)
    assert (
        engine_main(["measure", "--dets", str(f), "--labels", str(labels), "--out", str(out)]) == 0
    )
    assert json.loads(out.read_text(encoding="utf-8"))["ap50"] == pytest.approx(1.0)
