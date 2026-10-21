import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.io import imread_any
from baykus.motion.config import MotionCfg
from baykus.motion.ego import Ego
from baykus.motion.geometry import SIMILARITIES, corner_error, invert, similarity
from baykus.motion.images import half_gray

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "fixtures" / "m3ot" / "clip" / "sequences" / "clip"
CFG_PATH = ROOT / "configs" / "engine" / "motion.json"
W, H = 640, 512


@pytest.fixture(scope="module")
def cfg() -> MotionCfg:
    return MotionCfg.load(CFG_PATH)


def _ir(n: int) -> np.ndarray:
    return imread_any(CLIP / "ir" / f"{n:06d}.png", cv2.IMREAD_GRAYSCALE)


def _pan(f: int, d: tuple[float, float]) -> np.ndarray:
    """Kamera kayması: f. karede görüntü −d·(f − 1) kayar (harita = 1. kare)."""
    return np.array([[1, 0, -d[0] * (f - 1)], [0, 1, -d[1] * (f - 1)]], np.float64)


def _pan_frames(n: int, d: tuple[float, float]) -> dict[int, np.ndarray]:
    base = _ir(1)
    return {f: half_gray(cv2.warpAffine(base, _pan(f, d), (W, H))) for f in range(1, n + 1)}


def _run(ego: Ego, frames: dict[int, np.ndarray | None], source: str = "ir") -> dict:
    return {f: ego.update(g, source if g is not None else "none", [], f) for f, g in frames.items()}


# --- config -----------------------------------------------------------------------------------


def test_cfg_loads_repo_file_with_sha(cfg):
    assert cfg.ego.max_corners == 300 and cfg.reloc.library_max == 300
    assert cfg.motion.k == 15 and cfg.frozen.mad_max == 0.5 and cfg.log.max_events == 20000
    assert cfg.sha256 == hashlib.sha256(CFG_PATH.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutate", ["missing", "extra", "missing_section"])
def test_cfg_rejects_wrong_keys(tmp_path, mutate):
    raw = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if mutate == "missing":
        del raw["ego"]["lk_win"]
    elif mutate == "extra":
        raw["motion"]["typo"] = 1
    else:
        del raw["summary"]
    p = tmp_path / "m.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        MotionCfg.load(p)


# --- ego --------------------------------------------------------------------------------------


def test_first_frame_identity_gen1(cfg):
    r = Ego(cfg).update(half_gray(_ir(1)), "ir", [], 1)
    assert r.ok and r.map_gen == 1 and not r.provisional and r.keyframe
    assert np.allclose(r.frame_to_map, [[1, 0, 0], [0, 1, 0]])
    j = r.to_json()
    assert set(j) == {
        "ok", "source", "map_gen", "provisional", "frame_to_map", "inliers", "inlier_ratio",
        "keyframe", "relinked", "relocalized", "merge",
    }  # fmt: skip
    assert j["frame_to_map"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] and j["merge"] is None
    json.dumps(j, allow_nan=False)


def test_synthetic_similarity_recovered(cfg):
    # Kabul 3: 10 kare × 5 sabit benzerlik; köşe hatası ≤ 0,5 px, durumların ≥ %95'i.
    errs = []
    for n in range(1, 11):
        base = _ir(n)
        for tx, ty, deg, s in SIMILARITIES:
            t = similarity(tx, ty, deg, s, center=(W / 2, H / 2))
            ego = Ego(cfg)
            ego.update(half_gray(base), "ir", [], 1)
            r = ego.update(half_gray(cv2.warpAffine(base, t, (W, H))), "ir", [], 2)
            errs.append(corner_error(r.frame_to_map, invert(t), W, H) if r.ok else float("inf"))
    assert len(errs) == 50
    assert sum(e <= 0.5 for e in errs) >= 48, sorted(errs)[-5:]


def test_keyframe_rules_age(cfg):
    res = _run(Ego(cfg), _pan_frames(35, (0.6, 0.3)))
    assert [f for f, r in res.items() if r.keyframe] == [1, 31]  # 31. kare: yaş 30
    assert all(r.ok for r in res.values())


def test_keyframe_rules_shift(cfg):
    # 6 px/kare tam = 3 px/kare yarı: 14. kare 39 px (anahtar değil), 15. kare 42 px > 40.
    res = _run(Ego(cfg), _pan_frames(20, (6.0, 0.0)))
    assert [f for f, r in res.items() if r.keyframe] == [1, 15]
    assert corner_error(res[20].frame_to_map, invert(_pan(20, (6.0, 0.0))), W, H) <= 1.0


def test_failed_frame_never_keyframe(cfg):
    frames = _pan_frames(3, (1.0, 0.5))
    noise = np.random.default_rng(0).integers(0, 256, frames[2].shape, dtype=np.uint8)
    ego = Ego(cfg)
    r1 = ego.update(frames[1], "ir", [], 1)
    r2 = ego.update(noise, "ir", [], 2)
    r3 = ego.update(frames[3], "ir", [], 3)
    assert r1.keyframe and not r2.ok and not r2.keyframe and r2.frame_to_map is None
    assert r3.ok and r3.relinked and r3.map_gen == 1
    assert corner_error(r3.frame_to_map, invert(_pan(3, (1.0, 0.5))), W, H) <= 1.0


def test_short_gap_relinks_same_gen(cfg):
    frames: dict = _pan_frames(16, (1.0, 0.5))
    for f in range(11, 16):
        frames[f] = None
    res = _run(Ego(cfg), frames)
    assert all(not res[f].ok and res[f].map_gen == 1 for f in range(11, 16))
    r = res[16]
    assert r.ok and r.relinked and r.map_gen == 1 and not r.provisional
    assert corner_error(r.frame_to_map, invert(_pan(16, (1.0, 0.5))), W, H) <= 1.0


def _long_gap(cfg, ego: Ego | None = None, n_valid=10, lost=11):
    frames: dict = _pan_frames(n_valid + lost + 1, (1.0, 0.5))
    for f in range(n_valid + 1, n_valid + lost + 1):
        frames[f] = None
    ego = ego or Ego(cfg)
    return ego, _run(ego, frames)


def test_long_gap_opens_provisional(cfg):
    _, res = _long_gap(cfg)
    r = res[22]
    assert r.ok and r.map_gen == 2 and r.provisional and not r.relinked and r.keyframe
    assert np.allclose(r.frame_to_map, [[1, 0, 0], [0, 1, 0]])


def test_channel_change_opens_provisional(cfg):
    ego = Ego(cfg)
    g = half_gray(_ir(1))
    assert ego.update(g, "ir", [], 1).map_gen == 1
    r = ego.update(g, "rgb", [], 2)
    assert r.ok and r.map_gen == 2 and r.provisional and r.source == "rgb"


def test_provisional_becomes_permanent_after_window(cfg):
    ego, res = _long_gap(cfg)
    assert res[22].provisional
    g = half_gray(_ir(1))
    after = [ego.update(g, "ir", [], 22 + i) for i in range(1, cfg.reloc.window_frames + 1)]
    # Geçici neslin ilk 30 karesi (22..51) geçici; 30 kare sonra (52.) kalıcı.
    assert [r.provisional for r in after] == [True] * (cfg.reloc.window_frames - 1) + [False]
    assert all(r.map_gen == 2 and r.ok for r in after)
    assert ego.generations[2]["provisional"] is False


def test_break_inside_provisional_closes_it(cfg):
    ego, res = _long_gap(cfg)
    assert res[22].map_gen == 2 and res[22].provisional
    ego.break_continuity()
    r = ego.update(half_gray(_ir(3)), "ir", [], 23)
    assert r.map_gen == 3 and r.provisional
    assert ego.generations[2]["provisional"] is False


def test_break_continuity_opens_provisional(cfg):
    ego = Ego(cfg)
    g = half_gray(_ir(1))
    ego.update(g, "ir", [], 1)
    ego.break_continuity()
    r = ego.update(g, "ir", [], 2)
    assert r.map_gen == 2 and r.provisional


def test_ego_blank_frame_not_ok(cfg):
    ego = Ego(cfg)
    z = np.zeros((H // 2, W // 2), np.uint8)
    r = ego.update(z, "ir", [], 1)
    assert not r.ok and r.frame_to_map is None and r.map_gen == 1 and not r.keyframe
    g = half_gray(_ir(1))
    assert ego.update(g, "ir", [], 2).ok  # boş ilk kareden sonra ilk gerçek kare nesil 1
    for f in range(3, 7):
        r = ego.update(z, "ir", [], f)
        assert not r.ok and r.frame_to_map is None
    r = ego.update(g, "ir", [], 7)
    assert r.ok and r.map_gen == 2  # bağlama başarısız → kopma → geçici nesil
    json.dumps(r.to_json(), allow_nan=False)


def test_mask_excludes_boxes(cfg):
    box = (200.0, 150.0, 420.0, 330.0)
    ego = Ego(cfg)
    assert ego.update(half_gray(_ir(1)), "ir", [box], 1).ok
    pts = ego.key_points * 2 + 0.5  # tam çözünürlük
    assert len(pts) > 0
    d = cfg.ego.mask_dilate_px
    inside = (
        (pts[:, 0] >= box[0] - d)
        & (pts[:, 0] <= box[2] + d)
        & (pts[:, 1] >= box[1] - d)
        & (pts[:, 1] <= box[3] + d)
    )
    assert not inside.any()
    b = cfg.ego.border_px
    assert (pts[:, 0] >= b).all() and (pts[:, 0] <= W - b).all()
    assert (pts[:, 1] >= b).all() and (pts[:, 1] <= H - b).all()


def test_reset_starts_over(cfg):
    ego, _ = _long_gap(cfg)
    ego.reset()
    r = ego.update(half_gray(_ir(1)), "ir", [], 1)
    assert (
        r.map_gen == 1
        and not r.provisional
        and ego.generations == {1: {"provisional": False, "opened_n": 1}}
    )
