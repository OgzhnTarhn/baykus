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
    assert out.read_bytes() == expected.read_bytes()
