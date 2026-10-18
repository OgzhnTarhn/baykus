import json
import math

import cv2
import numpy as np
import pytest

from baykus.health.features import FEATURES, frame_features, misalign, misalign_vector
from baykus.health.gate import REASONS, assess, check_frame, decide_gate
from baykus.robustness.corruptions import shift
from tests.test_health_features import scene
from tests.test_health_gate import OK, TH


def _pair(dx: int = 20, dy: int = 0) -> tuple[np.ndarray, np.ndarray]:
    g = scene()
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), shift(g, dx, dy)


def test_vector_matches_shift():
    dx, dy, r = misalign_vector(*_pair(20, 0))
    assert abs(dx - 20) < 1.0 and abs(dy) < 1.0 and r > 0.1


def test_vector_sign_follows_ir_shift_down():
    dx, dy, _ = misalign_vector(*_pair(0, 12))
    assert abs(dx) < 1.0 and abs(dy - 12) < 1.0


def test_px_is_vector_norm():
    bgr, ir = _pair(14, -6)
    dx, dy, r = misalign_vector(bgr, ir)
    px, r2 = misalign(bgr, ir)
    assert px == pytest.approx(math.hypot(dx, dy), abs=1e-9) and r2 == r


def test_vector_none_when_camera_lost():
    bgr, _ = _pair()
    assert misalign_vector(bgr, np.zeros((512, 640), np.uint8)) is None


def test_features_csv_unchanged():
    bgr, ir = _pair()
    assert set(frame_features(bgr, ir)) == set(FEATURES)


def test_features_vector_adds_components():
    bgr, ir = _pair(20, 0)
    f = frame_features(bgr, ir, vector=True)
    assert set(f) == set(FEATURES) | {"misalign_dx", "misalign_dy"}
    assert abs(f["misalign_dx"] - 20) < 1.0
    assert f["misalign_px"] == pytest.approx(math.hypot(f["misalign_dx"], f["misalign_dy"]))
    lost = frame_features(bgr, np.zeros((512, 640), np.uint8), vector=True)
    assert lost["misalign_dx"] is None and lost["misalign_dy"] is None


def _phase2(f: dict) -> tuple[str, str, bool]:
    """Faz 2 kuralının bağımsız yeniden yazımı (öncelik: kayıp > gürültülü > hizasız)."""
    rgb_lost = f["rgb_std"] is not None and f["rgb_std"] <= TH.rgb_lost
    ir_lost = f["ir_std"] is not None and f["ir_std"] <= TH.ir_lost
    noisy = not ir_lost and f["ir_noise"] is not None and f["ir_noise"] >= TH.ir_noisy
    mis = (
        not (rgb_lost or ir_lost)
        and f["misalign_px"] is not None
        and f["misalign_px"] >= TH.misaligned
    )
    if rgb_lost and ir_lost:
        return "none", "iki kamera da kayıp", mis
    if rgb_lost:
        return "rgb_off", "RGB kayıp", mis
    if ir_lost:
        return "ir_off", "IR kayıp", mis
    if noisy:
        return "ir_off", "IR gürültülü", mis
    if mis:
        return "ir_off", "IR hizasız", mis
    return "none", "", mis


def test_r_min_none_keeps_phase2_values():
    rng = np.random.default_rng(7)
    for _ in range(20):
        f = {
            **OK,
            "rgb_std": float(rng.choice([0.0, 1.5, 40.0])),
            "ir_std": float(rng.choice([0.0, 2.0, 30.0])),
            "ir_noise": float(rng.uniform(0, 12)),
            "misalign_px": float(rng.uniform(0, 30)),
            "misalign_response": float(rng.uniform(0, 1)),
        }
        h = assess(f, TH)
        gate, reason, mis = _phase2(f)
        assert (h["gate"], h["reason"], h["pair"]["misaligned"]) == (gate, reason, mis)
        assert h["pair"]["misalign_px"] == f["misalign_px"]
        assert h["pair"]["response"] == f["misalign_response"]
        assert h["pair"]["misaligned_frame"] == mis


def test_r_min_removes_low_response():
    f = {**OK, "misalign_px": 30.0, "misalign_response": 0.05}
    assert assess(f, TH)["gate"] == "ir_off"
    h = assess(f, TH, r_min=0.2)
    assert h["gate"] == "none" and h["reason_code"] is None and h["reason"] == ""
    assert h["pair"]["misaligned_frame"] is False and h["pair"]["misaligned"] is False


def test_r_min_equal_keeps_alarm():
    h = assess({**OK, "misalign_px": 30.0, "misalign_response": 0.2}, TH, r_min=0.2)
    assert h["gate"] == "ir_off" and h["pair"]["misaligned"] is True


def test_r_min_with_missing_response_removes_alarm():
    h = assess({**OK, "misalign_px": 30.0, "misalign_response": None}, TH, r_min=0.2)
    assert h["gate"] == "none"


def test_reason_text_and_code():
    h = assess({**OK, "misalign_px": 20.0}, TH)
    assert h["reason_code"] == "misaligned" and h["reason"] == "IR hizasız"
    h = assess(OK, TH)
    assert h["reason_code"] is None and h["reason"] == ""
    json.dumps(h, allow_nan=False)


def test_pair_carries_vector_components():
    h = assess({**OK, "misalign_dx": 2.5, "misalign_dy": -1.0}, TH)
    assert (h["pair"]["dx_px"], h["pair"]["dy_px"]) == (2.5, -1.0)
    h = assess(OK, TH)  # Faz 2 özellik satırı: bileşen yok
    assert h["pair"]["dx_px"] is None and h["pair"]["dy_px"] is None


@pytest.mark.parametrize(
    "flags, expected",
    [
        ((True, True, False, False), ("none", "both_lost")),
        ((True, False, True, True), ("rgb_off", "rgb_lost")),
        ((False, True, False, True), ("ir_off", "ir_lost")),
        ((False, False, True, True), ("ir_off", "ir_noisy")),
        ((False, False, False, True), ("ir_off", "misaligned")),
        ((False, False, False, False), ("none", None)),
    ],
)
def test_decide_gate_priority(flags, expected):
    assert decide_gate(*flags) == expected
    if expected[1]:
        assert expected[1] in REASONS


def test_check_frame_passes_r_min_and_vector():
    bgr, ir = _pair(20, 0)
    _, _, h = check_frame(bgr, ir, TH)
    assert h["gate"] == "ir_off" and abs(h["pair"]["dx_px"] - 20) < 1.0
    r = h["pair"]["response"]
    _, ir_out, h2 = check_frame(bgr, ir, TH, r_min=r + 0.01)
    assert h2["gate"] == "none" and ir_out.any()
