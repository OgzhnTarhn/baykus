import json

import numpy as np

from baykus.data.cli import main
from baykus.data.verify import verify_split
from baykus.io import imwrite_any
from tests.conftest import make_raw_dataset


def test_all_good(raw_root):
    d = make_raw_dataset(raw_root, "train", ["00001", "00002"])
    ok, issues = verify_split(d, "train")
    assert ok == ["00001", "00002"]
    assert issues == []


def test_missing_ir_pair(raw_root):
    d = make_raw_dataset(raw_root, "train", ["00001", "00002"])
    (d.ir / "00002.jpg").unlink()
    ok, issues = verify_split(d, "train")
    assert ok == ["00001"]
    assert [(i.stem, i.reason) for i in issues] == [("00002", "missing_ir")]


def test_unreadable_and_size_mismatch(raw_root):
    d = make_raw_dataset(raw_root, "train", ["00001", "00002", "00003"])
    (d.rgb / "00001.jpg").write_bytes(b"not an image")
    imwrite_any(d.ir / "00002.jpg", np.zeros((10, 10, 3), np.uint8))
    ok, issues = verify_split(d, "train")
    reasons = {i.stem: i.reason for i in issues}
    assert ok == ["00003"]
    assert reasons == {"00001": "unreadable_rgb", "00002": "size_mismatch"}


def test_bad_xml(raw_root):
    d = make_raw_dataset(raw_root, "val", ["00001"])
    (d.label_ir / "00001.xml").write_text("<annotation><object>", encoding="utf-8")
    ok, issues = verify_split(d, "val")
    assert ok == [] and issues[0].reason == "bad_xml_ir"


def test_cli_writes_outputs(raw_root, tmp_path):
    for split in ("train", "val", "test"):
        make_raw_dataset(raw_root, split, ["00001"])
    d = raw_root / "train" / "trainimgr" / "00001.jpg"
    d.unlink()
    meta = tmp_path / "meta"
    assert main(["verify", "--raw", str(raw_root), "--meta", str(meta)]) == 0
    assert (meta / "excluded.txt").read_text(encoding="utf-8").strip() == "train/00001\tmissing_ir"
    ok = json.loads((meta / "ok_stems.json").read_text(encoding="utf-8"))
    assert ok == {"train": [], "val": ["00001"], "test": ["00001"]}
    rep = json.loads((meta / "verify_report.json").read_text(encoding="utf-8"))
    assert rep["issues_by_reason"] == {"missing_ir": 1}


def test_malformed_polygon_is_reported_not_raised(raw_root):
    d = make_raw_dataset(raw_root, "val", ["00001"])
    (d.label_rgb / "00001.xml").write_text(
        "<annotation><object><name>car</name><polygon><x1>1</x1><y1>1</y1></polygon>"
        "</object></annotation>",
        encoding="utf-8",
    )
    ok, issues = verify_split(d, "val")
    assert ok == [] and issues[0].reason == "bad_xml_rgb"
