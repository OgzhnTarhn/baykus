def test_cli_measure_oracle_and_test_guard(tmp_path):
    import json

    from baykus.motion.cli import main
    from baykus.motion.images import load_sequence
    from baykus.motion.labels import gt_tracks, label_sequence, write_labels

    from .motion_synth import make_video, write_sequence

    tracks = {1: {"start": (60, 60), "speed": (0, 0)}, 2: {"start": (150, 140), "speed": (3, 0)}}
    frames, gt = make_video(n=60, tracks=tracks)
    d = write_sequence(tmp_path / "data", "s1", frames, gt)
    seq = load_sequence(d, allow_test=False)
    tr = tmp_path / "data" / "labels" / "tracks_ir.csv"
    rows = label_sequence(seq, gt_tracks(tr, "s1"))
    write_labels(tmp_path / "motion" / "labels_val_s1.csv", rows)
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"split": "val", "sequences": ["s1"]}), encoding="utf-8")
    out = tmp_path / "o.json"
    args = ["--dev", str(dev), "--data", str(tmp_path / "data"), "--labels-dir",
            str(tmp_path / "motion"), "--out", str(out)]  # fmt: skip
    code = main(["measure", "--row", "oracle", *args])
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["gate_k9"]["pass"] is (code == 0)
    assert rec["labeled"]["n"] > 0 and rec["git_commit"] and rec["motion_sha256"]
    assert rec["labeled"]["R_mov"] > 0.5 and rec["labeled"]["R_sta"] > 0.5
    assert main(["measure", "--row", "oracle", "--split", "test", *args]) == 2  # bayraksız test


def test_motion_val_matches_committed_fixture(tmp_path):
    import json
    from pathlib import Path

    import pytest

    from baykus.motion.cli import main

    root = Path(__file__).resolve().parents[1]  # python/
    dets = root / "../fixtures/tracks/dev_dets.json.gz"
    tracks = root / "../fixtures/tracks/dev_tracks_py.json.gz"
    maps = root / "../fixtures/motion/maps_val.json.gz"
    motion = root / "../configs/engine/motion.json"
    expected = root / "../fixtures/expected/motion_val.json"
    for p in (dets, tracks, maps, motion, expected):
        if not p.exists():
            pytest.skip(f"depo fikstürü yok: {p}")
    out = tmp_path / "motion_val.json"
    code = main(["motion-val", "--dets", str(dets), "--tracks", str(tracks),
                 "--maps", str(maps), "--motion", str(motion), "--out", str(out)])  # fmt: skip
    assert code == 0
    got = json.loads(out.read_text(encoding="utf-8"))
    want = json.loads(expected.read_text(encoding="utf-8"))
    # bayt eşitliği değil: CI işlemcisine göre son ondalık hane değişebiliyor; büyük dosyada
    # pytest'in difflib farkı saatler sürüyordu (CI "takılması"); ilk farkı kısa yaz
    assert _first_diff(got, want) is None


def _first_diff(a, b, path="$"):
    import math

    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, int | float) and isinstance(b, int | float)
        return (
            None
            if ok and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
            else f"{path}: {a!r} != {b!r}"
        )
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: anahtarlar {sorted(a)} != {sorted(b)}"
        return next((d for k in a if (d := _first_diff(a[k], b[k], f"{path}.{k}"))), None)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: uzunluk {len(a)} != {len(b)}"
        return next(
            (
                d
                for i, (x, y) in enumerate(zip(a, b, strict=True))
                if (d := _first_diff(x, y, f"{path}[{i}]"))
            ),
            None,
        )
    return None if a == b else f"{path}: {a!r} != {b!r}"


_FM = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _parity_ego(frame_to_map=_FM, relocalized=False, merge=None) -> dict:
    return {
        "ok": True,
        "source": "ir",
        "map_gen": 1,
        "provisional": False,
        "frame_to_map": list(frame_to_map),
        "keyframe": False,
        "relinked": False,
        "relocalized": relocalized,
        "merge": merge,
    }


def _parity_frame(n: int, tracks: list[dict], ego: dict | None = None) -> dict:
    return {
        "frame": n,
        "reset_before": None,
        "tracks": tracks,
        "ego": _parity_ego() if ego is None else ego,
        "health": {
            "rgb": {"flags": [], "frozen": False},
            "ir": {"flags": [], "frozen": False},
            "pair": {"misaligned": False},
        },
    }


def _parity_reloc_case(dx: float) -> dict:
    clean = [_parity_frame(n, []) for n in (1, 2, 3)]
    moved = [_FM[0], _FM[1], _FM[2] + dx, _FM[3], _FM[4], _FM[5]]
    merge = {"from": 2, "to": 1, "transform": _FM}
    seek = [_parity_frame(1, []), _parity_frame(2, [], _parity_ego(moved, True, merge))]
    return {"scenarios": {"clean": clean, "seek": seek}}


def _parity_rows_case() -> dict:
    track = {"track_id": 1, "motion": "moving", "speed_rel": 0.5, "map_xy": [10.0, 20.0]}
    return {"scenarios": {"clean": [_parity_frame(i + 1, [dict(track)]) for i in range(3)]}}


def _parity_cli(tmp_path, ref: dict, cpp: dict, *extra: str) -> tuple[int, dict]:
    import json

    from baykus.motion.cli import main

    ref_p = tmp_path / "ref.json"
    cpp_p = tmp_path / "cpp.json"
    out = tmp_path / "parity.json"
    ref_p.write_text(json.dumps(ref), encoding="utf-8")
    cpp_p.write_text(json.dumps(cpp), encoding="utf-8")
    code = main(["parity", "--ref", str(ref_p), "--cpp", str(cpp_p),
                 "--out", str(out), *extra])  # fmt: skip
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_report_cli_writes_markdown(tmp_path):
    from baykus.motion.cli import main

    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "motion.md"
    code = main(["report", "--results", str(results), "--out", str(out)])
    assert code == 0
    assert out.is_file()
    assert "kanıt yok" in out.read_text(encoding="utf-8")


def test_parity_cli_known_fail_10b_exits_zero(tmp_path):
    import copy

    ref = _parity_reloc_case(2.5)
    code, rec = _parity_cli(
        tmp_path, ref, copy.deepcopy(ref), "--known-fail-10b", "docs/adr/0011-kabul-10b.md"
    )
    assert code == 0
    assert rec["known_fail"] == {
        "kabul10b": "docs/adr/0011-kabul-10b.md",
        "still_failing": True,
    }
    assert rec["kabul1"] == "pass"


def test_parity_cli_10b_fail_without_flag_exits_one(tmp_path):
    import copy

    ref = _parity_reloc_case(2.5)
    code, rec = _parity_cli(tmp_path, ref, copy.deepcopy(ref))
    assert code == 1
    assert rec["kabul10b"]["pass"] is False


def test_parity_cli_known_fail_does_not_hide_kabul1(tmp_path):
    import copy

    ref = _parity_rows_case()
    cpp = copy.deepcopy(ref)
    cpp["scenarios"]["clean"][1]["ego"]["keyframe"] = True
    code, rec = _parity_cli(tmp_path, ref, cpp, "--known-fail-10b", "docs/adr/0011-kabul-10b.md")
    assert code == 1
    assert rec["kabul1"] == "fail"


def test_parity_cli_known_fail_when_10b_passes(tmp_path):
    import copy

    ref = _parity_reloc_case(1.5)
    code, rec = _parity_cli(
        tmp_path, ref, copy.deepcopy(ref), "--known-fail-10b", "docs/adr/0011-kabul-10b.md"
    )
    assert code == 0
    assert rec["known_fail"]["still_failing"] is False
