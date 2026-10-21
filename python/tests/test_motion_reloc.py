import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.io import imread_any
from baykus.motion.config import MotionCfg
from baykus.motion.ego import Ego, ego_mask
from baykus.motion.geometry import compose, corner_error, invert, similarity, to_full
from baykus.motion.images import half_gray
from baykus.motion.reloc import Library, Record, Relocalizer, match, orb_features

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "fixtures" / "m3ot" / "clip" / "sequences" / "clip"
W, H = 640, 512
I2 = np.array([[1.0, 0, 0], [0, 1.0, 0]])


@pytest.fixture(scope="module")
def cfg() -> MotionCfg:
    return MotionCfg.load(ROOT / "configs" / "engine" / "motion.json")


def _ir(n: int) -> np.ndarray:
    return imread_any(CLIP / "ir" / f"{n:06d}.png", cv2.IMREAD_GRAYSCALE)


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return cv2.warpAffine(img, np.float64([[1, 0, dx], [0, 1, dy]]), (W, H))


def _pan(f: int, d=(1.0, 0.5)) -> np.ndarray:
    return np.array([[1, 0, -d[0] * (f - 1)], [0, 1, -d[1] * (f - 1)]], np.float64)


def _pan_frame(f: int, d=(1.0, 0.5)) -> np.ndarray:
    return half_gray(cv2.warpAffine(_ir(1), _pan(f, d), (W, H)))


def _rec(gen=1, frame=None, n=0, source="ir") -> Record:
    e = np.empty((0, 2), np.float32)
    return Record(gen, I2.copy(), frame, n, source, e, np.empty((0, 32), np.uint8))


def _record_of(cfg, gray_half, gen=1, m=I2, frame=1, n=1) -> Record:
    kps, desc = orb_features(gray_half, [], cfg)
    return Record(gen, m.copy(), frame, n, "ir", kps, desc)


# --- kütüphane --------------------------------------------------------------------------------


def test_library_cap_and_candidates_by_frame(cfg):
    lib = Library(cfg)
    for i in range(305):
        lib.add(_rec(gen=1, frame=i, n=i))
    lib.add(_rec(gen=2, frame=100, n=400))  # dışlanan nesil
    lib.add(_rec(gen=1, frame=100, n=401, source="rgb"))  # başka kanal
    assert len(lib) == cfg.reloc.library_max
    assert min(r.n for r in lib.records) == 7  # en eski 7 kayıt çıktı (305 + 2 − 300)
    got = lib.candidates("ir", 100, exclude_gen=2)
    assert [r.frame for r in got] == [100, 101, 99, 102, 98]  # eşitlikte daha yeni n
    # |Δframe| ≤ max_dframe dışındakiler aday değil; uzak frame → aday yok (yeni nesil doğal).
    assert lib.candidates("ir", 304 + 301, exclude_gen=2) == []
    assert [r.frame for r in lib.candidates("ir", 304 + 299, exclude_gen=2)] == [304, 303]


def test_candidates_fall_back_to_recent(cfg):
    lib = Library(cfg)
    for i in range(10):
        lib.add(_rec(gen=1, frame=None if i % 2 else i, n=i))
    lib.add(_rec(gen=3, n=50))
    assert [r.n for r in lib.candidates("ir", None, exclude_gen=3)] == [9, 8, 7, 6, 5]


def test_remap_rewrites_provisional_records(cfg):
    lib = Library(cfg)
    lib.add(_rec(gen=1, n=1))
    lib.add(dataclasses.replace(_rec(gen=2, n=2), m_key=similarity(3, 4, 0, 1)))
    t = similarity(10, -5, 0.5, 1.01)
    lib.remap_gen(2, 1, t)
    r1, r2 = lib.records
    assert r1.gen == 1 and np.allclose(r1.m_key, I2)
    assert r2.gen == 1 and np.allclose(r2.m_key, compose(t, similarity(3, 4, 0, 1)))
    lib.reset()
    assert len(lib) == 0


# --- eşleştirme -------------------------------------------------------------------------------


@pytest.mark.parametrize("dx", [60, 120, 200])
def test_match_recovers_large_shift(cfg, dx):
    # Kabul 10 (a): büyük öteleme → köşe hatası ≤ 1,0 px.
    for n in (1, 5, 10):
        base = _ir(n)
        rec = _record_of(cfg, half_gray(base))
        kps, desc = orb_features(half_gray(_shift(base, dx, 0)), [], cfg)
        a, inl, ratio = match(kps, desc, rec, cfg)
        assert a is not None, (n, inl, ratio)
        assert corner_error(to_full(a), similarity(dx, 0, 0, 1), W, H) <= 1.0


def test_orb_features_canonical_order(cfg):
    g = half_gray(_ir(1))
    kps, desc = orb_features(g, [], cfg)
    assert len(kps) > 0 and desc.shape == (len(kps), 32)
    # Noktalar (y, x) sırasında azalmayan.
    keys = [(float(y), float(x)) for x, y in kps]
    assert keys == sorted(keys)
    # Aynı görüntüde iki çağrı aynı diziyi verir.
    kps2, desc2 = orb_features(g, [], cfg)
    assert np.array_equal(kps, kps2) and np.array_equal(desc, desc2)
    # Tanımlayıcı satırları noktalarla eş: sıralanmış ham cv2 çıktısıyla küme olarak aynı.
    e = cfg.ego
    mask = ego_mask(g.shape[:2], [], e.mask_dilate_px, e.border_px)
    raw_kps, raw_desc = cv2.ORB_create(nfeatures=cfg.reloc.orb_features).detectAndCompute(g, mask)
    order = sorted(
        range(len(raw_kps)),
        key=lambda i: (
            raw_kps[i].pt[1],
            raw_kps[i].pt[0],
            raw_kps[i].octave,
            raw_kps[i].response,
            raw_kps[i].angle,
        ),
    )
    raw_pts = np.float32([raw_kps[i].pt for i in order]).reshape(-1, 2)
    assert np.array_equal(kps, raw_pts)
    assert np.array_equal(desc, raw_desc[order])


def test_reloc_blank_frame_no_merge(cfg):
    z = np.zeros((H // 2, W // 2), np.uint8)
    kps, desc = orb_features(z, [], cfg)
    assert len(kps) == 0 and desc.shape == (0, 32)
    rl = Relocalizer(cfg)
    rl.add_keyframe(half_gray(_ir(1)), [], gen=1, m_key=I2, frame=1, n=1, source="ir")
    assert rl.try_merge(z, [], "ir", 2, I2, prov_gen=2) is None
    assert match(kps, desc, rl.library.records[0], cfg)[0] is None


def test_mirrored_scene_never_merges(cfg):
    # Kabul 10 (c): yatay aynalanmış sahne benzerlikle eşleşemez.
    rl = Relocalizer(cfg)
    for n in range(1, 11):
        rl.add_keyframe(half_gray(_ir(n)), [], gen=1, m_key=I2, frame=n, n=n, source="ir")
    for n in range(1, 11):
        assert rl.try_merge(half_gray(cv2.flip(_ir(n), 1)), [], "ir", n, I2, prov_gen=2) is None


def test_merge_transform_maps_provisional_to_target(cfg):
    rl = Relocalizer(cfg)
    rl.add_keyframe(_pan_frame(1), [], gen=1, m_key=I2, frame=1, n=1, source="ir")
    m_prov = similarity(7, -3, 0.4, 1.01)  # geçerli karenin geçici nesildeki dönüşümü
    mg = rl.try_merge(_pan_frame(20), [], "ir", 20, m_prov, prov_gen=2)
    assert mg is not None and mg.from_gen == 2 and mg.to_gen == 1
    assert corner_error(compose(mg.transform, m_prov), invert(_pan(20)), W, H) <= 1.0


def test_merge_picks_most_inliers(cfg):
    rl = Relocalizer(cfg)
    rl.add_keyframe(_pan_frame(1), [], gen=1, m_key=I2, frame=1, n=1, source="ir")
    rl.add_keyframe(_pan_frame(150), [], gen=4, m_key=I2, frame=150, n=2, source="ir")
    mg = rl.try_merge(_pan_frame(3), [], "ir", None, I2, prov_gen=5)
    assert mg is not None and mg.to_gen == 1


# --- ego ile ----------------------------------------------------------------------------------


def _feed(ego: Ego, spec: list[tuple[int, int | None]]):
    """spec: (istek kare no, sentetik kare no | None = kayıp)."""
    out = []
    for frame, img in spec:
        g = None if img is None else _pan_frame(img)
        out.append(ego.update(g, "ir" if g is not None else "none", [], frame))
    return out


def test_ego_merges_after_long_gap(cfg):
    ego = Ego(cfg, Relocalizer(cfg))
    spec = [(f, f) for f in range(1, 11)] + [(f, None) for f in range(11, 22)]
    spec += [(f, f) for f in range(22, 30)]
    res = _feed(ego, spec)
    r = res[21]  # 22. kare: geçici nesil açılır ve ilk denemede birleşir
    assert r.ok and r.relocalized and r.merge is not None
    assert r.merge.from_gen == 2 and r.merge.to_gen == 1
    assert r.map_gen == 1 and not r.provisional
    assert corner_error(r.frame_to_map, invert(_pan(22)), W, H) <= 1.0
    for f, r in zip(range(23, 30), res[22:], strict=True):
        assert r.ok and r.map_gen == 1 and not r.relocalized and r.merge is None
        assert corner_error(r.frame_to_map, invert(_pan(f)), W, H) <= 1.0
    assert ego.generations[2]["merged_into"] == 1


def test_ego_seek_back_relocalizes(cfg):
    ego = Ego(cfg, Relocalizer(cfg))
    _feed(ego, [(f, f) for f in range(1, 41)])
    ego.break_continuity()
    res = _feed(ego, [(f, f) for f in range(10, 16)])
    assert res[0].relocalized and res[0].merge.to_gen == 1
    assert all(r.map_gen == 1 for r in res)
    assert corner_error(res[-1].frame_to_map, invert(_pan(15)), W, H) <= 1.0


def test_ego_scene_change_becomes_permanent(cfg):
    # Aynalanmış sahne: 30 karede birleşme yok → kalıcı yeni nesil; deneme ilk + her 5. ok karede.
    attempts: list[int | None] = []

    class Spy(Relocalizer):
        def try_merge(self, gray_half, boxes, source, frame, m_prov, prov_gen):
            attempts.append(frame)
            return super().try_merge(gray_half, boxes, source, frame, m_prov, prov_gen)

    ego = Ego(cfg, Spy(cfg))
    _feed(ego, [(f, f) for f in range(1, 21)])
    ego.break_continuity()
    res = []
    for f in range(21, 21 + cfg.reloc.window_frames + 2):
        g = half_gray(cv2.flip(cv2.warpAffine(_ir(1), _pan(f), (W, H)), 1))
        res.append(ego.update(g, "ir", [], f))
    assert all(r.map_gen == 2 and not r.relocalized for r in res)
    assert res[cfg.reloc.window_frames - 1].provisional and not res[-1].provisional
    assert attempts == [21, 26, 31, 36, 41, 46]


def test_reset_clears_library(cfg):
    ego = Ego(cfg, Relocalizer(cfg))
    _feed(ego, [(1, 1), (2, 2)])
    assert len(ego.reloc.library) == 1
    ego.reset()
    assert len(ego.reloc.library) == 0
