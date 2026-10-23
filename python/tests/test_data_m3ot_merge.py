import csv
import json
from pathlib import Path

import numpy as np
import pytest

from baykus.data import m3ot
from baykus.data.cli import main
from baykus.data.m3ot_merge import merge, relabel
from baykus.data.teacher import label_tracks
from baykus.engine_ref.pipeline import Det
from baykus.io import imwrite_any


def _row(track_id: int, x: float, vis: float = 1.0, frame: int = 1) -> dict:
    return {
        "split": "train",
        "modality": "ir",
        "seq": "s",
        "frame": frame,
        "stem": f"s_{frame:06d}",
        "track_id": track_id,
        "x": x,
        "y": 10,
        "w": 8,
        "h": 6,
        "flag": 1,
        "cls": 1,
        "vis": vis,
    }


def _layout(root: Path) -> Path:
    """İki izli bir kare (s_000001), kutusuz bir kare (s_000002), görünürlüğü düşük bir satır."""
    rows = [_row(1, 5), _row(2, 30), _row(3, 40, vis=0.1)]
    (root / "meta").mkdir(parents=True)
    with (root / "meta/m3ot_tracks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=m3ot.TRACK_FIELDS)
        w.writeheader()
        w.writerows(rows)
    for stem in ("s_000001", "s_000002"):
        imwrite_any(root / f"ir/images/train/{stem}.png", np.full((64, 64), 128, np.uint8))
        imwrite_any(root / f"rgb/images/train/{stem}.jpg", np.full((64, 64, 3), 128, np.uint8))
        (root / "labels_ir/train").mkdir(parents=True, exist_ok=True)
        (root / f"labels_ir/train/{stem}.txt").write_text("x\n", encoding="utf-8")
    return root


def _classes(path: Path) -> list[str]:
    return sorted(ln.split()[0] for ln in path.read_text(encoding="utf-8").splitlines())


def test_relabel_writes_track_class(tmp_path: Path):
    root = _layout(tmp_path / "m3ot")
    counts = relabel(root, "train", {"s:1": {"cls_id": 1}, "s:2": {"cls_id": 3}})
    assert _classes(root / "labels_ir/train/s_000001.txt") == ["1", "3"]
    assert (root / "labels_ir/train/s_000002.txt").read_text(encoding="utf-8") == ""
    assert counts["truck"] == 1 and counts["van"] == 1 and counts["car"] == 0


def test_relabel_all_car(tmp_path: Path):
    root = _layout(tmp_path / "m3ot")
    counts = relabel(root, "train", None)
    assert _classes(root / "labels_ir/train/s_000001.txt") == ["0", "0"]
    assert counts["car"] == 2


def test_relabel_box_format_matches_converter(tmp_path: Path):
    root = _layout(tmp_path / "m3ot")
    relabel(root, "train", None)
    want = m3ot.yolo_lines([m3ot.GtRow(1, 1, 5, 10, 8, 6, 1, 1, 1.0)], 64, 64)
    assert want[0] in (root / "labels_ir/train/s_000001.txt").read_text(encoding="utf-8")


def test_relabel_rejects_test_split(tmp_path: Path):
    with pytest.raises(ValueError):
        relabel(_layout(tmp_path / "m3ot"), "test", None)


def test_merge_copies_with_prefix(tmp_path: Path):
    src, dv = _layout(tmp_path / "m3ot"), tmp_path / "dv"
    assert merge(src, dv, "train") == 2
    for rel in (
        "rgb/images/train/m3ot_s_000001.jpg",
        "ir/images/train/m3ot_s_000001.png",
        "labels_ir/train/m3ot_s_000002.txt",
    ):
        assert (dv / rel).is_file(), rel


def test_merge_refuses_overwrite(tmp_path: Path):
    src, dv = _layout(tmp_path / "m3ot"), tmp_path / "dv"
    merge(src, dv, "train")
    with pytest.raises(ValueError):
        merge(src, dv, "train")


def test_merge_rejects_test_split(tmp_path: Path):
    with pytest.raises(ValueError):
        merge(_layout(tmp_path / "m3ot"), tmp_path / "dv", "test")


class _FakePipe:
    """Her pencerede kutunun tam üstünde, izin x'ine göre sınıf veren tespit."""

    def __init__(self):
        self.calls = 0

    def detect(self, bgr, ir, floor):
        self.calls += 1
        assert bgr.shape[:2] == ir.shape[:2]
        return {"dets_all": [Det(0, 3, 0.9, 0.9, [5.0, 10.0, 13.0, 16.0])]}


def test_label_tracks_end_to_end(tmp_path: Path):
    root = _layout(tmp_path / "m3ot")
    pipe = _FakePipe()
    out = label_tracks(root, "train", pipe)
    assert set(out) == {"s:1", "s:2"}  # görünürlüğü düşük iz yok
    assert out["s:1"]["cls_id"] == 3 and out["s:1"]["frames"] == [1]
    assert pipe.calls == 2


def test_teacher_cli_rejects_test_split_before_models(tmp_path: Path):
    with pytest.raises(ValueError):
        main(["m3ot-teacher", "--root", str(tmp_path), "--split", "test",
              "--models", str(tmp_path / "yok"), "--out", str(tmp_path / "t.json")])  # fmt: skip


def test_audit_page_then_score(tmp_path: Path):
    root = _layout(tmp_path / "m3ot")
    teacher = tmp_path / "teacher.json"
    tracks = {"s:1": {"cls_id": 1, "votes": [], "frames": [1]}, "s:2": {"cls_id": 0}}
    teacher.write_text(json.dumps({"tracks": tracks}), encoding="utf-8")
    page = tmp_path / "audit"
    assert (
        main(["m3ot-audit", "--root", str(root), "--teacher", str(teacher), "--out", str(page)])
        == 0
    )
    key = json.loads((page / "key.json").read_text(encoding="utf-8"))
    assert sorted(key.values()) == ["s:1", "s:2"] and (page / "index.html").is_file()
    truth = {"s:1": "truck", "s:2": "unknown"}
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({i: truth[k] for i, k in key.items()}), encoding="utf-8")
    out = tmp_path / "score.json"
    args = ["m3ot-audit-score", "--teacher", str(teacher), "--key", str(page / "key.json"),
            "--answers", str(answers), "--out", str(out)]  # fmt: skip
    assert main(args) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["accuracy"] == 1.0 and rec["answered"] == 1 and rec["decision"] == "teacher"
