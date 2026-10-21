from pathlib import Path

import numpy as np
import pytest

from baykus.motion.config import MotionCfg
from baykus.motion.ego import EgoResult, Merge
from baykus.motion.geometry import apply, compose, invert, similarity
from baykus.motion.state import MotionState, TrackMotion, TrackObs

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg() -> MotionCfg:
    return MotionCfg.load(ROOT / "configs" / "engine" / "motion.json")


def _ego(m: np.ndarray | None, gen: int = 1, provisional: bool = False, merge=None) -> EgoResult:
    return EgoResult(
        ok=m is not None,
        source="ir" if m is not None else "none",
        map_gen=gen,
        provisional=provisional,
        frame_to_map=None if m is None else np.asarray(m, np.float64),
        inliers=None,
        inlier_ratio=None,
        keyframe=False,
        relinked=False,
        relocalized=merge is not None,
        merge=merge,
    )


def _box(cx: float, cy: float, size: float = 10.0):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


S2 = similarity(0, 0, 0, 2.0)
I2 = similarity(0, 0, 0, 1.0)


def _run(st: MotionState, n: int, step: float, m=S2, tid: int = 1, x0: float = 50.0):
    """Kutu her karede `step` px sağa (kare koordinatında); `m` sabit harita dönüşümü."""
    return [st.update(_ego(m), [TrackObs(tid, _box(x0 + step * i, 40))])[tid] for i in range(n)]


def test_speed_in_map_units(cfg):
    # Ölçek 2, 1 px/kare, kutu 10: harita 2 px/kare, s = 10·2 = 20 → v = 2·15/15/20 = 0,10.
    out = _run(MotionState(cfg), 20, 1.0)
    assert all(o.motion == "unknown" and o.speed_rel is None for o in out[:15])
    assert out[15].speed_rel == pytest.approx(0.1) and out[15].motion == "moving"  # sınır dahil
    assert out[15].map_xy == pytest.approx((2 * (50 + 15), 2 * 40))


def test_stationary_boundary_inclusive(cfg):
    # 0,3 px/kare, ölçek 1, kutu 10 → v = 0,03.
    out = _run(MotionState(cfg), 20, 0.3, m=I2)
    assert out[15].speed_rel == pytest.approx(0.03) and out[15].motion == "stationary"


def test_hysteresis_band_keeps_previous(cfg):
    st = MotionState(cfg)
    out = _run(st, 20, 1.0)  # moving
    assert out[-1].motion == "moving"
    # 0,5 px/kare → harita 1 px/kare, v = 0,05 (bant): karar sürer.
    x = 50 + 19.0
    band = []
    for _ in range(40):
        x += 0.5
        band.append(st.update(_ego(S2), [TrackObs(1, _box(x, 40))])[1])
    assert all(o.motion == "moving" for o in band)
    assert band[-1].speed_rel == pytest.approx(0.05)  # pencere tamamen bantta
    # Yeni iz bantta başlarsa unknown.
    st2 = MotionState(cfg)
    assert _run(st2, 20, 0.5)[-1].motion == "unknown"


def test_new_track_unknown_first_k(cfg):
    st = MotionState(cfg)
    _run(st, 20, 1.0, tid=1)
    out = _run(st, 16, 1.0, tid=2, x0=200)
    assert [o.motion for o in out] == ["unknown"] * 15 + ["moving"]


def test_hold_then_unknown(cfg):
    st = MotionState(cfg)
    _run(st, 20, 1.0)
    held = [st.update(_ego(None), [TrackObs(1, _box(0, 0))])[1] for _ in range(17)]
    assert all(o.speed_rel is None and o.map_xy is None for o in held)
    assert [o.motion for o in held] == ["moving"] * cfg.motion.hold + ["unknown"] * 2


def test_ego_gap_in_window_blocks_v(cfg):
    st = MotionState(cfg)
    _run(st, 20, 1.0)
    st.update(_ego(None), [TrackObs(1, _box(0, 0))])  # tek kayıp karesi (n = 21)
    out = [st.update(_ego(S2), [TrackObs(1, _box(50 + 20 + i, 40))])[1] for i in range(17)]
    # n = 22..36: pencere kayıp kareyi içerir ya da t − K kaydı yok → v yok, karar tutulur.
    assert all(o.speed_rel is None for o in out[:15])
    assert out[15].speed_rel is not None  # n = 37: t − K = 22
    assert [o.motion for o in out[:14]] == ["moving"] * 14  # son v n = 20; 20 + 15 = 35
    assert out[14].motion == "unknown"  # n = 36: tutma bitti
    assert out[15].motion == "moving"


def test_track_absent_frames_still_measured(cfg):
    # İzin aradaki kareleri gerekmez: geçerli ve tam t − K karesinde kaydı yeter (ego penceresi ok).
    st = MotionState(cfg)
    _run(st, 20, 1.0)
    for _ in range(5):
        st.update(_ego(S2), [])  # iz görünmüyor (n = 21..25)
    o = st.update(_ego(S2), [TrackObs(1, _box(80, 40))])[1]  # n = 26, t − K = 11 (x = 60)
    assert o.motion == "moving" and o.speed_rel == pytest.approx(2 * 20 / 15 / 20)


def test_new_generation_resets(cfg):
    st = MotionState(cfg)
    _run(st, 20, 1.0)
    out = [
        st.update(_ego(S2, gen=2, provisional=True), [TrackObs(1, _box(70 + i, 40))])[1]
        for i in range(16)
    ]
    assert [o.motion for o in out] == ["unknown"] * 15 + ["moving"]
    assert out[0].speed_rel is None


def test_merge_carries_history(cfg):
    st = MotionState(cfg)
    m_prov = S2  # geçici nesil 2'de karenin dönüşümü
    t = similarity(30, -12, 0.0, 0.5)  # geçici → hedef
    out = []
    for i in range(20):
        out.append(
            st.update(_ego(m_prov, gen=2, provisional=True), [TrackObs(1, _box(50 + i, 40))])[1]
        )
    assert out[-1].motion == "moving"
    v_before = out[-1].speed_rel
    mg = Merge(2, 1, t)
    o = st.update(_ego(compose(t, m_prov), gen=1, merge=mg), [TrackObs(1, _box(70, 40))])[1]
    # Taşınan geçmişle v kesintisiz: c ve s aynı oranda ölçeklenir → v aynı.
    assert o.motion == "moving" and o.speed_rel == pytest.approx(v_before)
    assert o.map_xy == pytest.approx(tuple(apply(compose(t, m_prov), np.array([[70.0, 40.0]]))[0]))
    nxt = st.update(_ego(compose(t, m_prov), gen=1), [TrackObs(1, _box(71, 40))])[1]
    assert nxt.speed_rel == pytest.approx(v_before)


def test_frozen_frame_holds_motion(cfg):
    st = MotionState(cfg)
    _run(st, 20, 1.0)
    o = st.update(_ego(None), [TrackObs(1, _box(70, 40))])[1]  # donmuş kare: ego_ok yok
    assert o == TrackMotion("moving", None, None)


def test_reset_clears(cfg):
    st = MotionState(cfg)
    _run(st, 20, 1.0)
    st.reset()
    assert _run(st, 1, 1.0)[0].motion == "unknown"


def test_untracked_merge_from_other_gen_does_not_reset(cfg):
    # Birleşme geçici neslin ilk karesindeyse hareket modülü nesil 2'yi hiç görmez: geçmiş sürer.
    st = MotionState(cfg)
    _run(st, 20, 1.0)
    mg = Merge(2, 1, invert(I2))
    o = st.update(_ego(S2, gen=1, merge=mg), [TrackObs(1, _box(70, 40))])[1]
    assert o.motion == "moving" and o.speed_rel == pytest.approx(0.1)
