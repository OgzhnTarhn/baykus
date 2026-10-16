import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.data.cli import main as data_main
from baykus.data.m3ot import (
    GtRow,
    classify,
    convert_m3ot,
    frame_images,
    keep,
    read_gt,
    yolo_lines,
)
from baykus.data.multichannel import build_rgbt4
from baykus.io import imread_any, imwrite_any

LAYOUTS = {
    "dirs": lambda root, split, seq, mod: root / split / mod.upper() / seq,  # test/RGB/seqA
    "suffix": lambda root, split, seq, mod: root / split / f"{seq}_{mod}",  # test/seqA_ir
    "nested": lambda root, split, seq, mod: root / split / seq / mod,  # test/seqA/ir
}


def make_raw(root: Path, layout: str, extra_rgb_only_seq: bool = False) -> Path:
    """seqA: 3 ortak kare + sadece RGB'de 4. kare; seqB: 2 ortak kare. 40×32.
    Her karede bir geçerli kutu (iz 7) + 1. karede süzülecek iki satır (flag 0, görünürlük 0.2)."""
    seqs = [("seqA", [1, 2, 3]), ("seqB", [1, 2])]
    if extra_rgb_only_seq:
        seqs.append(("seqC", [1]))
    for seq, frames in seqs:
        for mod in ("rgb", "ir"):
            if seq == "seqC" and mod == "ir":
                continue
            d = LAYOUTS[layout](root, "test", seq, mod)
            extra = [4] if seq == "seqA" and mod == "rgb" else []
            for f in frames + extra:
                img = np.full(
                    (32, 40, 3), 60 if mod == "rgb" else 120, np.uint8
                )  # IR 3 kanal kopya
                imwrite_any(d / "img1" / f"{f:06d}.jpg", img)
            rows = [f"{f},7,4,6,10,8,1,1,1.0" for f in frames]
            rows += ["1,8,0,0,5,5,0,1,1.0", "1,9,20,20,6,6,1,1,0.2"]
            gt = d / "gt" / "gt.txt"
            gt.parent.mkdir(parents=True, exist_ok=True)
            gt.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("rel", "want"),
    [
        ("test/RGB/seqA", ("test", "rgb", "seqA")),
        ("test/seqA_ir", ("test", "ir", "seqA")),
        ("Test/seqA/infrared", ("test", "ir", "seqA")),
        ("m3ot/val/Seq-03-Visible", ("val", "rgb", "Seq_03")),
        ("M3OT/1/ir/test/1-03T", ("test", "ir", "1_03")),
        ("M3OT/2/rgb/test/2-03", ("test", "rgb", "2_03")),
    ],
)
def test_classify(rel, want):
    assert classify(Path(rel)) == want


def test_classify_errors():
    with pytest.raises(ValueError, match="okunamadı"):
        classify(Path("test/seqA"))  # modalite yok
    with pytest.raises(ValueError, match="okunamadı"):
        classify(Path("RGB/seqA"))  # bölüm yok


def test_read_gt_and_filter(tmp_path: Path):
    p = tmp_path / "gt.txt"
    p.write_text("1,7,4,6,10,8,1,1,1.0\n2,8,0,0,5,5,0,1,1.0\n3,9,1,1,5,5,1,2,1.0\n"
                 "4,10,1,1,5,5,1,1,0.25\n", encoding="utf-8")  # fmt: skip
    rows = read_gt(p)
    assert rows[0] == GtRow(1, 7, 4.0, 6.0, 10.0, 8.0, 1, 1, 1.0)
    assert [keep(r) for r in rows] == [True, False, False, False]  # flag 0, sınıf 2, vis ≤ 0.25


def test_read_gt_short_line_names_line(tmp_path: Path):
    p = tmp_path / "gt.txt"
    p.write_text("1,7,4,6,10,8,1,1,1.0\n2,8,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"gt\.txt:2"):
        read_gt(p)


def test_yolo_lines_clip_and_drop():
    rows = [
        GtRow(1, 7, 4, 6, 10, 8, 1, 1, 1.0),
        GtRow(1, 8, -4, 0, 10, 8, 1, 1, 1.0),  # sola taşıyor → x 0..6
        GtRow(1, 9, 45, 0, 5, 5, 1, 1, 1.0),  # tamamen dışarıda → atlanır
    ]
    assert yolo_lines(rows, 40, 32) == [
        "0 0.225000 0.312500 0.250000 0.250000",
        "0 0.075000 0.125000 0.150000 0.250000",
    ]


def test_frame_images(tmp_path: Path):
    for name in ("000002.jpg", "frame_000010.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert {k: v.name for k, v in frame_images(tmp_path).items()} == {
        2: "000002.jpg",
        10: "frame_000010.jpg",
    }


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_convert_layouts(tmp_path: Path, layout: str):
    raw = make_raw(tmp_path / "raw", layout)
    out = tmp_path / "m3ot"
    stats = convert_m3ot(raw, out, splits=["test"])
    assert stats["pairs"] == 5 and stats["sequences"] == 2
    assert stats["frames_rgb_only"] == 1 and stats["frames_ir_only"] == 0
    assert stats["boxes"] == {"rgb": 5, "ir": 5}
    assert stats["dropped_rows"] == {"rgb": 4, "ir": 4}  # dizi başına 2 süzülen satır
    stems = sorted(p.stem for p in (out / "rgb/images/test").glob("*.jpg"))
    assert stems == ["seqA_000001", "seqA_000002", "seqA_000003", "seqB_000001", "seqB_000002"]
    ir = imread_any(out / "ir/images/test/seqA_000001.png", cv2.IMREAD_UNCHANGED)
    assert ir.ndim == 2 and ir.shape == (32, 40)
    lab = (out / "labels_ir/test/seqA_000001.txt").read_text(encoding="utf-8")
    assert lab == "0 0.225000 0.312500 0.250000 0.250000\n"
    coco = json.loads((out / "coco/test_ir.json").read_text(encoding="utf-8"))
    assert coco["categories"] == [{"id": 1, "name": "vehicle"}]
    assert len(coco["images"]) == 5 and len(coco["annotations"]) == 5
    assert (out / "data_rgb.yaml").exists()


def test_convert_real_m3ot_layout(tmp_path: Path):
    raw = tmp_path / "M3OT"
    for mod, seq in (("rgb", "1-03"), ("ir", "1-03T")):
        d = raw / "1" / mod / "test" / seq
        for f in (1, 3):
            img = np.full((32, 40, 3), 60 if mod == "rgb" else 120, np.uint8)
            imwrite_any(d / "img1" / f"{f:06d}.PNG", img)
        gt = d / "gt" / "gt.txt"
        gt.parent.mkdir(parents=True, exist_ok=True)
        gt.write_text("1,7,4,6,10,8,1,1,1.0\n3,7,4,6,10,8,1,1,1.0\n", encoding="utf-8")
    out = tmp_path / "m3ot"
    stats = convert_m3ot(raw, out, splits=["test"])
    assert stats["pairs"] == 2 and stats["sequences"] == 1
    assert stats["unpaired_sequences"] == []
    stems = sorted(p.stem for p in (out / "rgb/images/test").glob("*.jpg"))
    assert stems == ["1_03_000001", "1_03_000003"]


def test_k5_index_and_tracks(tmp_path: Path):
    out = tmp_path / "m3ot"
    convert_m3ot(make_raw(tmp_path / "raw", "suffix"), out, splits=["test"])
    with (out / "meta/m3ot_index.csv").open(encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    assert index[0] == {"split": "test", "stem": "seqA_000001", "seq": "seqA", "frame": "1"}
    with (out / "meta/m3ot_tracks.csv").open(encoding="utf-8") as f:
        tracks = list(csv.DictReader(f))
    # süzülenler dahil bütün satırlar: (3 + 2) kare × 2 modalite + 2 dizi × 2 süzülen × 2 modalite
    assert len(tracks) == 18
    t = tracks[0]
    assert (t["modality"], t["seq"], t["frame"], t["track_id"], t["stem"]) == (
        "rgb",
        "seqA",
        "1",
        "7",
        "seqA_000001",
    )
    stats = json.loads((out / "meta/m3ot_stats.json").read_text(encoding="utf-8"))
    assert stats["pairs"] == 5


def test_unpaired_sequence_is_counted_not_fatal(tmp_path: Path):
    raw = make_raw(tmp_path / "raw", "dirs", extra_rgb_only_seq=True)
    stats = convert_m3ot(raw, tmp_path / "m3ot", splits=["test"])
    assert stats["unpaired_sequences"] == ["test/seqC/rgb"] and stats["pairs"] == 5


def test_no_pairs_is_error(tmp_path: Path):
    raw = make_raw(tmp_path / "raw", "dirs")
    with pytest.raises(ValueError, match="çifti bulunamadı"):
        convert_m3ot(raw, tmp_path / "m3ot", splits=["train"])


def test_parallel_matches_serial_and_rgbt4(tmp_path: Path):
    raw = make_raw(tmp_path / "raw", "nested")
    convert_m3ot(raw, tmp_path / "a", splits=["test"], workers=1)
    convert_m3ot(raw, tmp_path / "b", splits=["test"], workers=2)
    for p in sorted((tmp_path / "a/ir/images/test").glob("*.png")):
        np.testing.assert_array_equal(
            imread_any(p, cv2.IMREAD_UNCHANGED),
            imread_any(tmp_path / "b/ir/images/test" / p.name, cv2.IMREAD_UNCHANGED),
        )
    assert build_rgbt4(tmp_path / "a", splits=["test"]) == 5  # 4 kanal aynı yoldan


def test_cli(tmp_path: Path, capsys):
    raw = make_raw(tmp_path / "raw", "dirs")
    assert data_main(["m3ot", "--raw", str(raw), "--out", str(tmp_path / "o"),
                      "--splits", "test"]) == 0  # fmt: skip
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["pairs"] == 5
