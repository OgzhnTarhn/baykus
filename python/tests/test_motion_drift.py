import numpy as np
import pytest

from baykus.motion.drift import Trial, map_drift, reloc_summary, stationary_stability
from baykus.motion.geometry import apply, compose, invert, similarity
from baykus.motion.labels import LabelRow
from baykus.motion.measure import FrameOut

W, H = 64, 48


class TinyFrames:
    name, split = "s", "val"

    def __init__(self, n: int):
        self.frames = list(range(1, n + 1))

    def ir(self, n: int) -> np.ndarray:
        return np.zeros((H, W), np.uint8)


def _m(f: int) -> np.ndarray:
    """Kare → harita: kamera her karede (1, 0,5) px kayar, hafif dönme."""
    return similarity(1.0 * (f - 1), 0.5 * (f - 1), 0.01 * (f - 1), 1.0)


def _ego(m: np.ndarray | None, gen: int = 1) -> dict:
    return {
        "ok": m is not None,
        "map_gen": gen,
        "frame_to_map": None if m is None else [float(v) for v in m.reshape(-1)],
    }


def test_map_drift_zero_for_exact_transforms():
    frames = TinyFrames(200)
    run = [FrameOut(f, _ego(_m(f)), []) for f in frames.frames]
    calls = []

    def exact(a, b, mask_a, mask_b, ta, tb):
        calls.append((ta, tb))
        return np.vstack([compose(invert(_m(tb)), _m(ta)), [0, 0, 1]])

    d = map_drift(frames, run, {}, deltas=(15, 60, 150), step=10, register=exact)
    assert d["15"]["pairs"] == 19 and d["150"]["pairs"] == 5  # t = 1, 11, ..., t + Δ ≤ 200
    for k in ("15", "60", "150"):
        assert d[k]["median_px"] == pytest.approx(0.0, abs=1e-9)
        assert d[k]["p95_px"] == pytest.approx(0.0, abs=1e-9)
        assert d[k]["center_median_px"] == pytest.approx(0.0, abs=1e-9)


def test_map_drift_measures_offset_and_skips_other_gen():
    frames = TinyFrames(40)
    gen = {f: 1 if f <= 20 else 2 if f <= 30 else 3 for f in frames.frames}
    run = [FrameOut(f, _ego(_m(f), gen=gen[f]), []) for f in frames.frames]
    shift = np.array([[1, 0, 3.0], [0, 1, 4.0], [0, 0, 1]])

    def off(a, b, mask_a, mask_b, ta, tb):
        return shift @ np.vstack([compose(invert(_m(tb)), _m(ta)), [0, 0, 1]])

    d = map_drift(frames, run, {}, deltas=(15,), step=10, register=off)
    assert d["15"]["pairs"] == 1 and d["15"]["skipped_gen"] == 2  # (1,16); (11,26), (21,36) değil
    assert d["15"]["median_px"] == pytest.approx(5.0)
    assert d["15"]["center_median_px"] == pytest.approx(5.0)
    assert d["15"]["center_p95_px"] == pytest.approx(5.0)


def test_map_drift_center_separate_from_corners():
    frames = TinyFrames(20)
    run = [FrameOut(f, _ego(_m(f)), []) for f in frames.frames]

    def off(a, b, mask_a, mask_b, ta, tb):
        p = compose(invert(_m(tb)), _m(ta))
        c = apply(p, np.array([[W / 2, H / 2]]))[0]
        zoom = similarity(0, 0, 0, 1.1, center=(c[0], c[1]))  # merkez sabit, köşeler kayar
        return np.vstack([compose(zoom, p), [0, 0, 1]])

    d = map_drift(frames, run, {}, deltas=(15,), step=10, register=off)["15"]
    assert d["center_median_px"] == pytest.approx(0.0, abs=1e-9)
    assert d["median_px"] == pytest.approx(0.1 * np.hypot(W / 2, H / 2))


def test_stability_segments_split_on_moves():
    labels, run = [], []
    rng = np.random.default_rng(0)
    for f in range(1, 101):
        lab = "moving" if 41 <= f <= 50 else "stationary"
        labels.append(LabelRow("s", f, 1, None, lab))
        x = 100.0 if f <= 40 else 300.0  # iki ayrı duruş
        jitter = rng.normal(0, 0.5, 2) if f <= 40 else rng.normal(0, 2.0, 2)
        box = (x + jitter[0] - 5, 50 + jitter[1] - 5, x + jitter[0] + 5, 50 + jitter[1] + 5)
        run.append(FrameOut(f, _ego(np.eye(2, 3)), [
            {"track_id": 1, "box": box, "motion": "stationary",
             "map_xy": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)}]))  # fmt: skip
    labels.append(LabelRow("s", 150, 1, None, "stationary"))  # 1 karelik bölüm: < 30 sayılmaz
    st = stationary_stability(labels, run)
    assert st["segments"] == 2 and st["skipped"] == 0
    r1, r2 = st["ratios"]
    assert r1 < r2  # ikinci duruşun titremesi daha büyük
    assert r1 == pytest.approx(np.sqrt(2) * 0.5 / 10, rel=0.35)


def test_reloc_report_counts_false_links():
    trials = [
        Trial("s", 100, "back", True, 1, 0.4),
        Trial("s", 100, "forward", True, 2, 12.0),  # yapay yanlış bağlama
        Trial("s", 200, "back", False, 6, None),
        Trial("t", 100, "back", True, 1, None),  # hedef nesil kesintisiz koşudakinden farklı
    ]
    r = reloc_summary(trials)
    assert r["trials"] == 4 and r["success"] == 3 and r["success_rate"] == 0.75
    assert r["false_links"] == 1 and r["compared"] == 2
    assert r["diff_px"]["median"] == pytest.approx(6.2)
    assert r["attempts"]["median"] == 1.5
    assert r["per_seq"]["s"]["trials"] == 3


def test_reloc_trials_on_panning_clip():
    from pathlib import Path

    import cv2

    from baykus.io import imread_any
    from baykus.motion.config import MotionCfg
    from baykus.motion.drift import reloc_trials

    root = Path(__file__).resolve().parents[2]
    cfg = MotionCfg.load(root / "configs" / "engine" / "motion.json")
    base = imread_any(
        root / "fixtures/m3ot/clip/sequences/clip/ir/000001.png", cv2.IMREAD_GRAYSCALE
    )

    class Pan:
        name, split, frames = "pan", "val", list(range(1, 61))

        def ir(self, f):
            m = np.float64([[1, 0, -1.0 * (f - 1)], [0, 1, -0.5 * (f - 1)]])
            return cv2.warpAffine(base, m, (640, 512))

    trials = reloc_trials(Pan(), {}, cfg, every=20, jump=10)
    assert [(t.at, t.direction) for t in trials] == [
        (20, "back"), (20, "forward"), (40, "back"), (40, "forward"), (60, "back")]  # fmt: skip
    assert all(t.success and t.attempts == 1 for t in trials)
    assert all(t.diff_px is not None and t.diff_px <= 1.0 for t in trials)
