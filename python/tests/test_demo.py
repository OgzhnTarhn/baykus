import csv
import json
from pathlib import Path

import numpy as np
import pytest

from baykus.data.cli import main as data_main
from baykus.data.demo import consecutive_sequences, pack, pack_labels, pack_sequence
from baykus.data.m3ot import TRACK_FIELDS
from baykus.io import imwrite_any

W, H = 40, 32


def make_processed(root: Path, split: str, seqs: dict[str, list[int]], ir_size=(W, H)) -> Path:
    """baykus-data m3ot çıktı düzeni: görüntüler, meta/m3ot_index.csv, coco, m3ot_tracks.csv."""
    index, tracks, images, anns = [], [], [], []
    for seq, frames in seqs.items():
        for f in frames:
            stem = f"{seq}_{f:06d}"
            rgb = np.full((H, W, 3), f, np.uint8)
            imwrite_any(root / "rgb/images" / split / f"{stem}.jpg", rgb)
            imwrite_any(
                root / "ir/images" / split / f"{stem}.png",
                np.full((ir_size[1], ir_size[0]), 100 + f, np.uint8),
            )
            index.append([split, stem, seq, f])
            img_id = len(images) + 1
            images.append({"id": img_id, "file_name": stem, "width": W, "height": H})
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": 1,
                         "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0})  # fmt: skip
            for mod in ("rgb", "ir"):
                tracks.append([split, mod, seq, f, stem, 7, 1, 2, 3, 4, 1, 1, 1.0])
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "m3ot_index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "stem", "seq", "frame"])
        w.writerows(index)
    with (meta / "m3ot_tracks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(TRACK_FIELDS)
        w.writerows(tracks)
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "vehicle"}]}
    (root / "coco").mkdir(parents=True, exist_ok=True)
    (root / "coco" / f"{split}_ir.json").write_text(json.dumps(coco), encoding="utf-8")
    return root


def test_consecutive_detects_gap(tmp_path):
    root = make_processed(tmp_path, "val", {"a": [1, 2, 3], "b": [1, 2, 4]})
    got = consecutive_sequences(root / "meta/m3ot_index.csv", "val")
    assert got == {
        "a": {"frames": 3, "consecutive": True},
        "b": {"frames": 3, "consecutive": False},
    }


def test_consecutive_ignores_other_splits(tmp_path):
    root = make_processed(tmp_path, "val", {"a": [1, 2]})
    assert consecutive_sequences(root / "meta/m3ot_index.csv", "train") == {}


def test_pack_copies_bytes_and_writes_index(tmp_path):
    root = make_processed(tmp_path / "src", "val", {"a": [1, 2, 3]})
    out = tmp_path / "out"
    seq_dir = pack_sequence(root, "val", "a", out)
    assert seq_dir == out / "sequences/a"
    for f in (1, 2, 3):
        src_rgb = (root / "rgb/images/val" / f"a_{f:06d}.jpg").read_bytes()
        src_ir = (root / "ir/images/val" / f"a_{f:06d}.png").read_bytes()
        assert (seq_dir / "rgb" / f"{f:06d}.jpg").read_bytes() == src_rgb
        assert (seq_dir / "ir" / f"{f:06d}.png").read_bytes() == src_ir
    index = json.loads((seq_dir / "index.json").read_text("utf-8"))
    assert index == {
        "name": "a",
        "split": "val",
        "frames": [1, 2, 3],
        "fps": 30,
        "size": [W, H],
        "source": "M3OT val, CC BY 4.0",
    }


def test_pack_refuses_non_consecutive(tmp_path):
    root = make_processed(tmp_path / "src", "val", {"gappy": [1, 2, 4]})
    with pytest.raises(ValueError, match="gappy"):
        pack_sequence(root, "val", "gappy", tmp_path / "out")


def test_pack_refuses_unknown_sequence(tmp_path):
    root = make_processed(tmp_path / "src", "val", {"a": [1, 2]})
    with pytest.raises(ValueError, match="yok"):
        pack_sequence(root, "val", "zzz", tmp_path / "out")


def test_pack_refuses_size_mismatch(tmp_path):
    root = make_processed(tmp_path / "src", "val", {"a": [1, 2]}, ir_size=(W + 2, H))
    with pytest.raises(ValueError, match="boyut"):
        pack_sequence(root, "val", "a", tmp_path / "out")


def test_pack_refuses_test_without_flag(tmp_path):
    root = make_processed(tmp_path / "src", "test", {"a": [1, 2]})
    with pytest.raises(ValueError, match="test"):
        pack(root, "test", ["a"], tmp_path / "out")
    pack(root, "test", ["a"], tmp_path / "out", allow_test=True)
    assert (tmp_path / "out/sequences/a/index.json").exists()


def test_pack_labels_filters_sequences(tmp_path):
    root = make_processed(tmp_path / "src", "val", {"a": [1, 2], "b": [1]})
    out = tmp_path / "out"
    pack_labels(root, "val", ["a"], out)
    coco = json.loads((out / "labels/val_ir.json").read_text("utf-8"))
    assert [im["file_name"] for im in coco["images"]] == ["a_000001", "a_000002"]
    kept = {im["id"] for im in coco["images"]}
    assert coco["annotations"] and all(a["image_id"] in kept for a in coco["annotations"])
    assert coco["categories"] == [{"id": 1, "name": "vehicle"}]
    with (out / "labels/tracks_ir.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {(r["seq"], r["modality"]) for r in rows} == {("a", "ir")}
    assert len(rows) == 2


def test_cli_seqs_and_demo_pack(tmp_path, capsys):
    root = make_processed(tmp_path / "src", "val", {"a": [1, 2], "b": [1, 3]})
    assert data_main(["m3ot-seqs", "--root", str(root), "--split", "val"]) == 0
    assert json.loads(capsys.readouterr().out)["b"]["consecutive"] is False
    out = tmp_path / "out"
    rc = data_main(
        ["demo-pack", "--root", str(root), "--split", "val", "--seqs", "a", "--out", str(out)]
    )
    assert rc == 0
    assert (out / "sequences/a/index.json").exists() and (out / "labels/val_ir.json").exists()
