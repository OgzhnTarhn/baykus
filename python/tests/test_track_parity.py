from pathlib import Path

from baykus.track.parity import compare, evidence, verdict


def _seq(ids_per_frame: list[list[tuple[int, int]]]) -> dict:
    return {
        "frames": [
            {"frame": f + 1, "tracks": [{"index": i, "track_id": t} for i, t in fr]}
            for f, fr in enumerate(ids_per_frame)
        ]
    }


def test_compare_identity_permutation_ok():
    py = _seq([[(0, -1), (1, -1)], [(0, 1), (1, 2)], [(0, 1), (1, 2)]])
    cpp = _seq([[(1, -1), (0, -1)], [(0, 7), (1, 3)], [(1, 3), (0, 7)]])
    r = compare({"a": py}, {"a": cpp})["a"]
    assert r == {"frames": 3, "compared": 3, "first_diff": None, "ratio": 1.0}


def test_compare_stops_at_first_diff():
    py = _seq([[(0, 1)], [(0, 1)], [(0, 1)], [(0, 1)]])
    cpp = _seq([[(0, 5)], [(0, 6)], [(0, 6)], [(0, 6)]])  # 2. karede 1 → 6: eşleme bozuldu
    r = compare({"a": py}, {"a": cpp})["a"]
    assert r == {"frames": 4, "compared": 1, "first_diff": 2, "ratio": 0.25}


def test_compare_minus_one_only_with_minus_one():
    py = _seq([[(0, -1)]])
    cpp = _seq([[(0, 4)]])
    assert compare({"a": py}, {"a": cpp})["a"]["first_diff"] == 1


def test_compare_detects_missing_index():
    py = _seq([[(0, 1), (1, 2)]])
    cpp = _seq([[(0, 1)]])
    assert compare({"a": py}, {"a": cpp})["a"]["first_diff"] == 1


def test_compare_bijection_both_ways():
    py = _seq([[(0, 1), (1, 2)]])
    cpp = _seq([[(0, 5), (1, 5)]])  # iki Python kimliği tek C++ kimliğine
    assert compare({"a": py}, {"a": cpp})["a"]["first_diff"] == 1


def _dump(iou, assign, scores=(0.9,)):
    return {
        "dets": [{"index": i, "score_raw": s} for i, s in enumerate(scores)],
        "iou1": iou,
        "assign1": assign,
        "iou2": [],
        "assign2": [],
    }


def test_evidence_iou_boundary():
    a = _dump([[0.1 + 5e-6, 0.5]], [[0, 1]])
    assert evidence(a, a) == "iou_boundary"
    b = _dump([[0.1 + 2e-5, 0.5]], [[0, 1]])
    assert evidence(b, b) is None


def test_evidence_score_boundary():
    a = _dump([[0.5]], [[0, 0]], scores=(0.6 + 5e-7,))
    assert evidence(a, a) == "score_boundary"
    assert evidence(_dump([[0.5]], [[0, 0]], scores=(0.7 - 5e-7,)), a) == "score_boundary"


def test_evidence_equal_cost_assignment():
    iou = [[0.5, 0.5], [0.5, 0.5]]
    py, cpp = _dump(iou, [[0, 0], [1, 1]], (0.9, 0.9)), _dump(iou, [[0, 1], [1, 0]], (0.9, 0.9))
    assert evidence(py, cpp) == "equal_cost_assignment"
    worse = _dump([[0.5, 0.4], [0.4, 0.5]], [[0, 1], [1, 0]], (0.9, 0.9))
    assert evidence(_dump([[0.5, 0.4], [0.4, 0.5]], [[0, 0], [1, 1]], (0.9, 0.9)), worse) is None


def test_evidence_invalid_assignment_is_not_evidence():
    iou = [[0.5, 0.5], [0.5, 0.5]]
    py, cpp = _dump(iou, [[0, 0], [1, 1]], (0.9, 0.9)), _dump(iou, [[0, 0], [1, 0]], (0.9, 0.9))
    assert evidence(py, cpp) is None


def _cmp(first_diff, ratio):
    return {
        "a": {"frames": 100, "compared": int(ratio * 100), "first_diff": first_diff, "ratio": ratio}
    }


def test_verdict_pass():
    assert verdict(_cmp(None, 1.0), {}) == "pass"


def test_verdict_ratio_rule():
    assert verdict(_cmp(86, 0.85), {"a": "iou_boundary"}) == "fail"  # ilk fark %85'te
    assert verdict(_cmp(91, 0.90), {"a": "iou_boundary"}) == "pass_with_exception"
    assert verdict(_cmp(91, 0.90), {"a": None}) == "fail"  # kanıtsız fark


def _dets_file(tmp_path, split="val"):
    import gzip
    import json

    frames = [
        {
            "frame": f,
            "dets": [
                {"index": 0, "cls_id": 0, "score_raw": 0.9, "score": 0.9,
                 "box": [10 + 4 * f, 10, 40 + 4 * f, 30]},
                {"index": 1, "cls_id": 0, "score_raw": 0.05, "score": 0.0,
                 "box": [300, 300, 310, 310]},
            ],
        }
        for f in range(1, 6)
    ]  # fmt: skip
    p = tmp_path / f"{split}.json.gz"
    p.write_bytes(
        gzip.compress(json.dumps({"seq": "1_08", "split": split, "frames": frames}).encode())
    )
    return p


def test_cli_pack_run_compare_self(tmp_path):
    import gzip
    import json

    from baykus.track.cli import main

    cfg = str(Path(__file__).resolve().parents[2] / "configs/engine/tracker.json")
    pack = tmp_path / "dev_dets.json.gz"
    assert main(["pack-dets", "--dets", str(_dets_file(tmp_path)), "--out", str(pack)]) == 0
    rec = json.loads(gzip.decompress(pack.read_bytes()))
    assert all(len(fr["dets"]) == 1 for fr in rec["sequences"][0]["frames"])  # 0,05 süzüldü
    tr = tmp_path / "py.json.gz"
    assert main(["run", "--dets", str(pack), "--out", str(tr), "--config", cfg, "--dump", "3"]) == 0
    assert "3" in json.loads(gzip.decompress(tr.read_bytes()))["sequences"][0]["dumps"]
    out = tmp_path / "parity.json"
    assert main(["compare", "--py", str(tr), "--cpp", str(tr), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "pass"


def test_cli_refuses_test_split(tmp_path):
    from baykus.track.cli import main

    assert main(["pack-dets", "--dets", str(_dets_file(tmp_path, "test")), "--out", "x.gz"]) == 2
    cfg = str(Path(__file__).resolve().parents[2] / "configs/engine/tracker.json")
    t = str(_dets_file(tmp_path, "test"))
    assert main(["run", "--dets", t, "--out", str(tmp_path / "o.gz"), "--config", cfg]) == 2


def test_cli_oracle_writes_sequences_and_refuses_test(tmp_path):
    import gzip
    import json

    from baykus.track.cli import main

    csv = tmp_path / "tracks_ir.csv"
    csv.write_text(
        "split,modality,seq,frame,stem,track_id,x,y,w,h,flag,cls,vis\n"
        "val,ir,1_08,2,1_08_000002,5,10,20,30,40,1,1,1.0\n",
        encoding="utf-8",
    )
    for split in ("val", "test"):
        d = tmp_path / split / "1_08"
        d.mkdir(parents=True)
        idx = {"name": "1_08", "split": split, "frames": [1, 2]}
        (d / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    out = tmp_path / "oracle.json.gz"
    args = ["oracle", "--gt-csv", str(csv), "--out", str(out), "--seq-dir"]
    assert main([*args, str(tmp_path / "val" / "1_08")]) == 0
    rec = json.loads(gzip.decompress(out.read_bytes()))
    (s,) = rec["sequences"]
    assert s["seq"] == "1_08" and s["split"] == "val" and rec["source"] == "oracle"
    assert [len(fr["dets"]) for fr in s["frames"]] == [0, 1]
    assert main([*args, str(tmp_path / "test" / "1_08")]) == 2
    assert main([*args, str(tmp_path / "test" / "1_08"), "--i-know-this-is-test"]) == 0
