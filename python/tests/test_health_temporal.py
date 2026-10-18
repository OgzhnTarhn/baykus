import copy

import pytest

from baykus.health.gate import assess
from baykus.health.temporal import MisalignWindow
from tests.test_health_gate import OK, TH


def _h(dx: float, dy: float = 0.0, flagged: bool = True, **over) -> dict:
    """Kare kuralı sonucu: flagged iken misalign_px eşiğin üstünde (Faz 2 alarmı)."""
    px = 30.0 if flagged else 3.0
    return assess(
        {**OK, "misalign_px": px, "misalign_dx": dx, "misalign_dy": dy, **over}, TH, r_min=0.2
    )


def _run(frames: list[dict], **kw) -> list[bool]:
    return [h["pair"]["misaligned"] for h in MisalignWindow(**kw).sync_pairs(frames)]


def test_constant_shift_alarms_from_third_frame():
    out = MisalignWindow().sync_pairs([_h(20, 0) for _ in range(5)])
    assert [h["pair"]["misaligned"] for h in out] == [False, False, True, True, True]
    assert [h["gate"] for h in out] == ["none", "none", "ir_off", "ir_off", "ir_off"]


def test_jumping_estimates_never_alarm():
    vs = [(40, 0), (0, 40), (-40, 0), (0, -40), (30, 30)]
    assert _run([_h(dx, dy) for dx, dy in vs]) == [False] * 5


def test_boundary_inclusive():
    # S = {(0,0), (0,0), (3,0)}: ortanca (0,0), üçüncünün uzaklığı tam 3.0 → C'de
    assert _run([_h(0), _h(0), _h(3)]) == [False, False, True]
    assert _run([_h(0), _h(0), _h(3.0000001)]) == [False, False, False]


def test_even_S_median_is_mean_of_middle():
    # S = 4 kare, x: 0, 0, 6, 6 → ortanca 3 (numpy.median); hepsi tam 3 px uzakta → |C| = 4
    assert _run([_h(0), _h(0), _h(6), _h(6)]) == [False, False, False, True]


def test_unflagged_frames_are_not_in_S():
    frames = [_h(20), _h(20, flagged=False), _h(20)]
    assert _run(frames) == [False, False, False]  # |S| = 2


def test_window_drops_old_frames():
    frames = [_h(20), _h(20), _h(20)] + [_h(20, flagged=False)] * 3
    # 6. karede pencere (son 5) yalnız 2 işaretli kare tutar
    assert _run(frames) == [False, False, True, True, True, False]


def test_lost_overrides():
    frames = [_h(20), _h(20), _h(20, ir_std=0.0)]
    out = MisalignWindow().sync_pairs(frames)[-1]
    assert out["pair"]["misaligned"] is False
    assert (out["gate"], out["reason_code"], out["reason"]) == ("ir_off", "ir_lost", "IR kayıp")


def test_noisy_has_priority():
    frames = [_h(20), _h(20), _h(20, ir_noise=12.0)]
    out = MisalignWindow().sync_pairs(frames)[-1]
    assert out["gate"] == "ir_off" and out["reason_code"] == "ir_noisy"


def test_reason_text_after_temporal():
    out = MisalignWindow().sync_pairs([_h(20) for _ in range(3)])[-1]
    assert out["reason_code"] == "misaligned" and out["reason"] == "IR hizasız"


def test_single_frame_alarm_is_suppressed():
    out = MisalignWindow().update(_h(20))
    assert out["pair"]["misaligned_frame"] is True and out["pair"]["misaligned"] is False
    assert out["gate"] == "none" and out["reason_code"] is None and out["reason"] == ""


def test_reset_clears():
    w = MisalignWindow()
    for _ in range(3):
        w.update(_h(20))
    w.reset()
    assert w.update(_h(20))["pair"]["misaligned"] is False


def test_can_alarm_on_unflagged_frame():
    frames = [_h(20), _h(20), _h(20), _h(20, flagged=False)]
    assert _run(frames) == [False, False, True, True]


def test_update_does_not_mutate_input():
    h = _h(20)
    before = copy.deepcopy(h)
    MisalignWindow().update(h)
    assert h == before


def test_missing_vector_is_error():
    h = assess({**OK, "misalign_px": 30.0}, TH)  # dx_px None ama kare işaretli
    with pytest.raises(ValueError, match="dx_px"):
        MisalignWindow().update(h)


def test_params_from_constructor():
    assert _run([_h(20), _h(20)], min_frames=2) == [False, True]
