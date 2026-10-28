import gzip
import json
import os
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.engine_ref.pipeline import Pipeline
from baykus.io import imread_any, imwrite_any
from baykus.motion import fixtures as fx
from baykus.motion.config import MotionCfg
from baykus.motion.ego import Ego
from baykus.motion.geometry import SIMILARITIES, similarity
from baykus.motion.images import half_gray
from baykus.motion.reloc import Relocalizer
from baykus.track.runner import load_cfg
from tests.test_engine_ref import _models
from tests.test_motion_session import ScriptBackend

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "fixtures" / "m3ot" / "clip" / "sequences" / "clip"
W, H = 640, 512
N = 62  # 60 karelik klip + iki başlangıç payı
CAM = 6.0  # px / kare: 60 karede ≥ 3 anahtar kare (yarı çözünürlük öteleme > 40 px)


@pytest.fixture(scope="module")
def mcfg() -> MotionCfg:
    return MotionCfg.load(ROOT / "configs" / "engine" / "motion.json")


@pytest.fixture(scope="module")
def dev_root(tmp_path_factory) -> Path:
    """Gerçek bir M3OT karesi 2× büyütülüp kayan pencereyle kırpılır (saf öteleme, kenar yok)."""
    root = tmp_path_factory.mktemp("dev")
    ir0 = imread_any(CLIP / "ir" / "000001.png", cv2.IMREAD_GRAYSCALE)
    rgb0 = imread_any(CLIP / "rgb" / "000001.jpg", cv2.IMREAD_COLOR)
    big_ir = cv2.resize(ir0, (2 * W, 2 * H), interpolation=cv2.INTER_LINEAR)
    big_rgb = cv2.resize(rgb0, (2 * W, 2 * H), interpolation=cv2.INTER_LINEAR)
    d = root / "sequences" / fx.SEQ
    for f in range(1, N + 1):
        x, y = round(CAM * (f - 1)), 200
        imwrite_any(d / "ir" / f"{f:06d}.png", big_ir[y : y + H, x : x + W])
        imwrite_any(d / "rgb" / f"{f:06d}.jpg", big_rgb[y : y + H, x : x + W])
    idx = {"name": fx.SEQ, "split": "val", "frames": list(range(1, N + 1)), "fps": 30}
    (d / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def stage(dev_root, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("stage")
    fx.write_clip(dev_root / "sequences" / fx.SEQ, 1, out)
    return out


def test_boundary_rule_shifts_start():
    bad = {1: ["ego kare 3: inliers 31 eşiğe 30 2'den yakın"], 2: ["clean: anahtar kare 2 < 3"]}
    start, notes = fx.pick_start([1, 2, 3, 4], lambda s: bad.get(s, []))
    assert start == 3
    assert len(notes) == 2 and notes[0].startswith("başlangıç 1:") and "inliers 31" in notes[0]
    with pytest.raises(ValueError, match="sınır kuralı"):
        fx.pick_start([1, 2], lambda s: bad[s])


def test_best_start_full_coverage_then_fewest_issues():
    table = {
        5: (["a"], ["ir_drop: birleşme yok"]),  # en az sorun ama kapsam eksik
        6: (["a", "b", "c"], []),
        7: (["a", "b"], []),
        8: (["a", "b"], []),  # eşitlikte önceki başlangıç
    }
    start, issues, notes = fx.pick_best([5, 6, 7, 8], lambda s: table[s])
    assert (start, issues) == (7, ["a", "b"])
    assert notes[0] == "başlangıç 5: 1 sorun, kapsam eksik (ir_drop: birleşme yok)"
    assert notes[1] == "başlangıç 6: 3 sorun"
    with pytest.raises(ValueError, match="kapsam"):
        fx.pick_best([5], lambda s: table[s])


def test_track_coverage_needs_rows_and_both_decisions():
    def runs(motions: list[str]) -> dict:
        frames = [{"tracks": [{"motion": m}]} for m in motions]
        return {"clean": {"frames": frames}}

    assert fx.track_coverage(runs(["moving"] * 3 + ["stationary"] * 3), min_rows=6) == []
    short = fx.track_coverage(runs(["moving"] * 3 + ["stationary"] * 2), min_rows=6)
    assert short == ["clean: iz × kare 5 < 6"]
    one = fx.track_coverage(runs(["moving"] * 6), min_rows=6)
    assert one == ["clean: 'stationary' kararı yok"]


def test_margin_records_parse_where():
    recs = fx.margin_records(["clean kare 33: ego ratio 0.4878 eşiğe (0.5) 0.02'den yakın"])
    assert recs == [
        {"scenario": "clean", "frame": 33, "text": "ego ratio 0.4878 eşiğe (0.5) 0.02'den yakın"}
    ]


def test_ego_margins_only_where_decision_flips(mcfg):
    e = mcfg.ego
    far = dict(inliers=100, ratio=0.9, shift=10.0, relink=False, age=1)
    assert fx.ego_margins(mcfg, **far) == []
    assert fx.ego_margins(mcfg, **{**far, "inliers": e.ok_min_inliers + 1, "ratio": 0.45})
    assert fx.ego_margins(mcfg, **{**far, "ratio": e.key_min_ratio + 0.01})
    assert fx.ego_margins(mcfg, **{**far, "shift": e.key_max_shift_px - 0.5})
    assert fx.ego_margins(mcfg, **{**far, "inliers": e.ok_min_inliers + 2, "ratio": 0.45}) == []
    # kayıt zaten ok değilse anahtar kare eşikleri karara girmez
    assert fx.ego_margins(mcfg, **{**far, "ratio": 0.2, "shift": e.key_max_shift_px}) == []
    # yaş kuralı anahtar kareyi zaten zorluyorsa öteleme eşiği karara girmez
    assert fx.ego_margins(mcfg, **{**far, "shift": e.key_max_shift_px, "age": e.key_max_age}) == []


def test_reloc_margins_candidates_and_choice(mcfg):
    r = mcfg.reloc
    assert fx.reloc_margins(mcfg, [(200, 0.8), (10, 0.05)]) == []
    assert fx.reloc_margins(mcfg, [(r.min_inliers + 1, 0.8)])
    assert fx.reloc_margins(mcfg, [(200, r.min_ratio - 0.01)])
    assert fx.reloc_margins(mcfg, [(200, 0.8), (199, 0.7)])  # en iyi aday 2 noktadan yakın
    assert fx.reloc_margins(mcfg, [(200, 0.8), (198, 0.7)]) == []


def test_scenarios_layout():
    sc = fx.scenarios()
    assert list(sc) == list(fx.SCENARIOS)
    assert [s["frame"] for s in sc["clean"]] == list(range(1, 61))
    drop = [s["frame"] for s in sc["ir_drop"] if s["ir"] == "zero_ir.png"]
    assert drop == list(range(25, 37))
    assert all(s["rgb"].startswith("clip/") for s in sc["ir_drop"])
    frz = [s["frame"] for s in sc["ir_freeze"] if s["ir"] == "clip/ir/000019.png"]
    assert frz == list(range(19, 30))  # 19 kendisi + 20-29 tekrar
    gap = [s["frame"] for s in sc["short_gap"] if s["rgb"] == "zero_rgb.png"]
    assert gap == list(range(40, 45))
    gap = [s["frame"] for s in sc["long_gap"] if s["ir"] == "zero_ir.png"]
    assert gap == list(range(30, 46))
    seek = sc["seek"]
    assert [s["frame"] for s in seek] == list(range(1, 41)) + list(range(10, 61))
    assert [i for i, s in enumerate(seek) if s["reset_before"]] == [40]
    assert seek[40]["reset_before"] == "tracks"
    mir = [s["frame"] for s in sc["scene_change"] if s["ir"].startswith("mirror/")]
    assert mir == list(range(21, 61))
    assert [(s["frame"], s["reset_before"]) for s in sc["reset_all"] if s["reset_before"]] == [
        (40, "all")
    ]


def test_clip_files(stage, dev_root):
    src = dev_root / "sequences" / fx.SEQ
    assert (stage / "clip" / "rgb" / "000001.jpg").read_bytes() == (
        src / "rgb" / "000001.jpg"
    ).read_bytes()
    assert (stage / "clip" / "ir" / "000060.png").read_bytes() == (
        src / "ir" / "000060.png"
    ).read_bytes()
    idx = json.loads((stage / "clip" / "index.json").read_text(encoding="utf-8"))
    assert idx["frames"] == list(range(1, 61)) and idx["split"] == "val"
    ir21 = imread_any(src / "ir" / "000021.png", cv2.IMREAD_GRAYSCALE)
    mir = imread_any(stage / "mirror" / "ir" / "000021.png", cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(mir, cv2.flip(ir21, 1))
    assert not (stage / "mirror" / "ir" / "000020.png").exists()
    assert not imread_any(stage / "zero_ir.png", cv2.IMREAD_GRAYSCALE).any()
    assert imread_any(stage / "zero_rgb.png", cv2.IMREAD_COLOR).shape == (H, W, 3)


def test_warps_known_transforms(stage):
    items = fx.write_warps(stage)
    assert len(items) == 80
    assert sum(i["kind"] == "small" for i in items) == 50
    assert sorted({i["src_frame"] for i in items}) == list(range(1, 61, 6))
    small = [i for i in items if i["kind"] == "small" and i["src_frame"] == 1]
    t = similarity(*SIMILARITIES[1], center=(W / 2, H / 2))
    assert np.allclose(np.array(small[1]["matrix"]).reshape(2, 3), t)
    big = [i for i in items if i["kind"] == "large"]
    assert sorted({i["params"]["tx"] for i in big}) == [60.0, 120.0, 200.0]
    src = imread_any(stage / "clip" / "ir" / "000001.png", cv2.IMREAD_GRAYSCALE)
    got = imread_any(stage / small[1]["path"], cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(got, cv2.warpAffine(src, t, (W, H)))
    listed = json.loads((stage / "warps.json").read_text(encoding="utf-8"))
    assert listed["warps"] == items


def test_scenarios_cover_generations(stage, mcfg):
    pipe = Pipeline(_models(), backend=ScriptBackend([np.zeros((1, 300, 6), np.float32)]))
    tcfg = load_cfg(ROOT / "configs" / "engine" / "tracker.json")
    runs = fx.run_scenarios(pipe, tcfg, mcfg, stage, fx.scenarios(), probe=True)
    ego = {k: [f["ego"] for f in v["frames"]] for k, v in runs.items()}

    assert sum(e["keyframe"] for e in ego["clean"][1:]) >= 3
    assert {e["map_gen"] for e in ego["clean"]} == {1}

    merges = [e["merge"] for e in ego["ir_drop"] if e["merge"]]
    assert [m["to"] for m in merges] == [1] and ego["ir_drop"][-1]["map_gen"] == 1

    assert ego["short_gap"][44]["relinked"]  # 45. kare: 5 karelik kayıptan dönüş
    assert {e["map_gen"] for e in ego["short_gap"]} == {1}

    for name in ("long_gap", "seek"):
        rel = [e for e in ego[name] if e["relocalized"]]
        assert len(rel) == 1 and rel[0]["merge"]["to"] == 1, name
        assert ego[name][-1]["map_gen"] == 1 and not ego[name][-1]["provisional"], name

    sc = ego["scene_change"]
    assert not any(e["merge"] for e in sc)
    assert sc[-1]["map_gen"] > 1 and not sc[-1]["provisional"]

    frozen = [f["frame"] for f in runs["ir_freeze"]["frames"] if f["health"]["ir"]["frozen"]]
    assert frozen == list(range(20, 30))

    after = runs["reset_all"]["frames"][39]
    assert after["frame"] == 40 and after["ego"]["keyframe"] and after["ego"]["map_gen"] == 1

    assert fx.coverage(runs) == []
    assert all("alarms" in f for v in runs.values() for f in v["frames"])


def test_tar_is_deterministic(stage, tmp_path):
    a, b = tmp_path / "a.tar", tmp_path / "b.tar"
    sha_a, n_a = fx.write_tar(stage, a)
    os.utime(stage / "zero_ir.png", (1e9, 1e9))
    sha_b, n_b = fx.write_tar(stage, b)
    assert a.read_bytes() == b.read_bytes() and sha_a == sha_b and n_a == n_b == a.stat().st_size
    with tarfile.open(a) as t:
        names = t.getnames()
        assert names == sorted(names) and "clip/index.json" in names
        assert {m.mtime for m in t.getmembers()} == {0}


def test_maps_val_shape():
    maps = json.loads(gzip.decompress((ROOT / "fixtures/motion/maps_val.json.gz").read_bytes()))
    dets = json.loads(gzip.decompress((ROOT / "fixtures/tracks/dev_dets.json.gz").read_bytes()))
    assert [s["seq"] for s in maps["sequences"]] == ["1_08", "2_08"]
    assert maps["motion_sha256"] == MotionCfg.load(ROOT / "configs/engine/motion.json").sha256
    for s, d in zip(maps["sequences"], dets["sequences"], strict=True):
        assert [f["frame"] for f in s["frames"]] == [f["frame"] for f in d["frames"]]
        assert len(s["frames"]) == 600
        for f in s["frames"]:
            e = f["ego"]
            assert e["source"] == "ir" and isinstance(e["map_gen"], int)
            assert (e["frame_to_map"] is None) == (not e["ok"])
            assert e["frame_to_map"] is None or len(e["frame_to_map"]) == 6
        assert sum(f["ego"]["ok"] for f in s["frames"]) > 500


def test_maps_val_masks_only_confident_dets(dev_root, mcfg, monkeypatch):
    # Maske takip katmanından yalnız ham skoru ego.mask_min_score'u geçenler (SessionRef gibi).
    dets = {
        "sequences": [
            {
                "seq": fx.SEQ,
                "split": "val",
                "frames": [
                    {
                        "frame": 1,
                        "dets": [
                            {"box": [10, 10, 20, 20], "score_raw": 0.3},
                            {"box": [30, 30, 40, 40], "score_raw": 0.9},
                        ],
                    }
                ],
            }
        ]
    }
    seen: list[list] = []
    orig = Ego.update

    def spy(self, gray, source, boxes, frame):
        seen.append(list(boxes))
        return orig(self, gray, source, boxes, frame)

    monkeypatch.setattr(fx.Ego, "update", spy)
    maps = fx.maps_val(dev_root, dets, mcfg)
    assert len(seen) == 1
    assert seen[0] == [(30, 30, 40, 40)]
    assert [f["frame"] for f in maps["sequences"][0]["frames"]] == [1]


def test_motion_val_matches_reference(mcfg):
    maps = json.loads(gzip.decompress((ROOT / "fixtures/motion/maps_val.json.gz").read_bytes()))
    dets = json.loads(gzip.decompress((ROOT / "fixtures/tracks/dev_dets.json.gz").read_bytes()))
    tracks = json.loads(
        gzip.decompress((ROOT / "fixtures/tracks/dev_tracks_py.json.gz").read_bytes())
    )
    want = json.loads((ROOT / "fixtures/expected/motion_val.json").read_text(encoding="utf-8"))
    got = fx.motion_expected(maps, dets, tracks, mcfg)
    assert got["sequences"] == want["sequences"]
    rows = [t for s in got["sequences"] for f in s["frames"] for t in f["tracks"]]
    assert {r[1] for r in rows} == {"moving", "stationary", "unknown"}


def test_ego_trace_replays_exactly(stage, mcfg):
    pipe = Pipeline(_models(), backend=ScriptBackend([np.zeros((1, 300, 6), np.float32)]))
    tcfg = load_cfg(ROOT / "configs" / "engine" / "tracker.json")
    scen = fx.scenarios()
    trace = fx.ego_trace(pipe, tcfg, mcfg, stage, scen)
    assert trace["motion_sha256"] == mcfg.sha256
    assert list(trace["scenarios"]) == list(scen)
    for name, steps in scen.items():
        rows = trace["scenarios"][name]
        assert len(rows) == len(steps)
        for rec in rows:
            if rec["source"] == "none":
                assert rec["ir"] is None
    for name, rows in trace["scenarios"].items():
        ego = Ego(mcfg, Relocalizer(mcfg))
        for rec in rows:
            if rec["reset_before"] == "all":
                ego.reset()
            elif rec["reset_before"] == "tracks":
                ego.break_continuity()
            gray = (
                None
                if rec["ir"] is None
                else half_gray(imread_any(stage / rec["ir"], cv2.IMREAD_GRAYSCALE))
            )
            r = ego.update(gray, rec["source"], [tuple(b) for b in rec["boxes"]], rec["frame"])
            assert r.to_json() == rec["ego"], (name, rec["frame"])
