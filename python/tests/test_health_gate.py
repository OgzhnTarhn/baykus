import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.health.gate import Thresholds, apply_gate, assess, check_frame, load_thresholds
from tests.test_health_features import scene

TH = Thresholds(
    rgb_lost=2,
    ir_lost=2,
    ir_noisy=8,
    misaligned=14,
    rgb_blurry=50,
    ir_blurry=50,
    dark=48.66,
    sha256="x",
)
OK = {
    "rgb_std": 40.0,
    "rgb_noise": 3.0,
    "rgb_blur": 300.0,
    "rgb_brightness": 120.0,
    "ir_std": 30.0,
    "ir_noise": 3.0,
    "ir_blur": 200.0,
    "misalign_px": 3.0,
    "misalign_response": 0.3,
}


@pytest.mark.parametrize(
    "over, gate, reason, rgb_state, ir_state",
    [
        ({}, "none", "", "ok", "ok"),
        ({"ir_noise": 8.0}, "ir_off", "IR gürültülü", "ok", "noisy"),  # sınır dahil
        ({"misalign_px": 20.0}, "ir_off", "IR hizasız", "ok", "ok"),
        ({"ir_noise": 12.0, "misalign_px": 20.0}, "ir_off", "IR gürültülü", "ok", "noisy"),
        ({"ir_std": 0.0, "misalign_px": None}, "ir_off", "IR kayıp", "ok", "lost"),
        ({"rgb_std": 0.0, "misalign_px": None}, "rgb_off", "RGB kayıp", "lost", "ok"),
        (
            {"rgb_std": 0.0, "ir_std": 0.0, "misalign_px": None},
            "none",
            "iki kamera da kayıp",
            "lost",
            "lost",
        ),
        (
            {"rgb_std": 0.0, "ir_noise": 12.0, "misalign_px": None},
            "rgb_off",
            "RGB kayıp",
            "lost",
            "noisy",
        ),
        ({"rgb_brightness": 20.0}, "none", "", "dark", "ok"),
        ({"rgb_blur": 10.0, "ir_blur": 10.0}, "none", "", "blurry", "blurry"),
        ({"misalign_px": None}, "none", "", "ok", "ok"),
    ],
)
def test_assess_states_and_gate(over, gate, reason, rgb_state, ir_state):
    h = assess({**OK, **over}, TH)
    got = (h["gate"], h["reason"], h["rgb"]["state"], h["ir"]["state"])
    assert got == (gate, reason, rgb_state, ir_state)
    json.dumps(h, allow_nan=False)


def test_flags_in_priority_order():
    h = assess({**OK, "rgb_brightness": 20.0, "rgb_blur": 10.0, "rgb_std": 0.0}, TH)
    assert h["rgb"]["flags"] == ["lost", "dark", "blurry"]


def test_misaligned_flag_in_pair():
    assert assess({**OK, "misalign_px": 14.0}, TH)["pair"]["misaligned"] is True
    assert assess(OK, TH)["pair"] == {
        "misalign_px": 3.0,
        "dx_px": None,
        "dy_px": None,
        "response": 0.3,
        "misaligned_frame": False,
        "misaligned": False,
    }
    assert assess(OK, TH)["reason_code"] is None


def test_misalignment_not_judged_when_a_camera_is_lost():
    h = assess({**OK, "ir_std": 0.0, "misalign_px": 30.0}, TH)
    assert h["pair"]["misaligned"] is False and h["reason"] == "IR kayıp"


def test_apply_gate_zeroes_one_channel_without_mutating():
    bgr, ir = np.full((4, 4, 3), 9, np.uint8), np.full((4, 4), 7, np.uint8)
    b2, i2 = apply_gate(bgr, ir, "ir_off")
    assert b2.tolist() == bgr.tolist() and not i2.any() and ir.all()
    b3, i3 = apply_gate(bgr, ir, "rgb_off")
    assert not b3.any() and i3.tolist() == ir.tolist() and bgr.all()
    with pytest.raises(ValueError):
        apply_gate(bgr, ir, "both_off")


def test_check_frame_changed_false_when_channel_already_zero():
    bgr = cv2.cvtColor(scene(), cv2.COLOR_GRAY2BGR)
    _, ir_out, h = check_frame(bgr, np.zeros((512, 640), np.uint8), TH)
    assert h["gate"] == "ir_off" and h["changed"] is False and not ir_out.any()


def test_check_frame_changed_true_when_gate_zeroes_pixels():
    g = scene()
    noisy_th = Thresholds(**{**TH.__dict__, "ir_noisy": 0.0})  # her IR "gürültülü"
    _, ir_out, h = check_frame(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 255 - g, noisy_th)
    assert h["gate"] == "ir_off" and h["changed"] is True and not ir_out.any()


def _fit_json(**over) -> dict:
    states = {
        "rgb_lost": 2.0,
        "ir_lost": 2.0,
        "ir_noisy": 8.0,
        "misaligned": 14.0,
        "rgb_blurry": 50.0,
        "ir_blurry": 50.0,
        **over,
    }
    return {"states": {k: {"threshold": v} for k, v in states.items()}, "dark_brightness": 48.66}


def test_load_thresholds(tmp_path: Path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(_fit_json(ir_noisy=9.5)), encoding="utf-8")
    th = load_thresholds(p)
    assert th.ir_noisy == 9.5 and th.dark == 48.66 and th.misaligned == 14.0
    assert th.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()


def test_load_thresholds_missing_state_is_error(tmp_path: Path):
    raw = _fit_json()
    del raw["states"]["misaligned"]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="misaligned"):
        load_thresholds(p)
