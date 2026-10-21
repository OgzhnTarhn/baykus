from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.health.frozen import FrozenDetector
from baykus.io import imread_any
from baykus.motion.config import MotionCfg
from baykus.motion.images import half_gray

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "fixtures" / "m3ot" / "clip" / "sequences" / "clip"


@pytest.fixture(scope="module")
def cfg() -> MotionCfg:
    return MotionCfg.load(ROOT / "configs" / "engine" / "motion.json")


def _ir(n: int) -> np.ndarray:
    return half_gray(imread_any(CLIP / "ir" / f"{n:06d}.png", cv2.IMREAD_GRAYSCALE))


def _rgb(n: int) -> np.ndarray:
    return half_gray(imread_any(CLIP / "rgb" / f"{n:06d}.jpg", cv2.IMREAD_COLOR))


def test_first_frame_has_no_diff(cfg):
    out = FrozenDetector(cfg).update(_rgb(1), _ir(1), False, False)
    for ch in ("rgb", "ir"):
        assert out[ch] == {"frozen": False, "diff_mad": None, "alarm": False}


def test_same_frame_is_frozen(cfg):
    fd = FrozenDetector(cfg)
    fd.update(_rgb(1), _ir(1), False, False)
    out = fd.update(_rgb(2), _ir(1), False, False)
    assert out["ir"]["frozen"] and out["ir"]["diff_mad"] == 0.0
    assert not out["rgb"]["frozen"] and out["rgb"]["diff_mad"] > 0.5


def test_consecutive_val_frames_not_frozen(cfg):
    fd = FrozenDetector(cfg)
    outs = [fd.update(_rgb(n), _ir(n), False, False) for n in range(1, 16)]
    assert not any(o[ch]["frozen"] or o[ch]["alarm"] for o in outs for ch in ("rgb", "ir"))


def test_threshold_inclusive(cfg):
    fd = FrozenDetector(cfg)
    a = np.zeros((4, 4), np.uint8) + 100
    b = a.copy()
    b[:2] += 1  # ortalama mutlak fark 0,5
    fd.update(a, a, False, False)
    out = fd.update(b, b, False, False)
    assert out["ir"]["diff_mad"] == 0.5 and out["ir"]["frozen"]


def test_alarm_three_of_five(cfg):
    fd = FrozenDetector(cfg)
    seq = [1, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8]  # 3., 5. ve 6. çağrı donmuş
    alarms = [fd.update(_rgb(n), _ir(n), False, False)["ir"]["alarm"] for n in seq]
    # Son 5 çağrıda ≥ 3 donmuş: 6. ve 7. çağrı.
    assert alarms == [False] * 5 + [True, True] + [False] * 4


def test_lost_is_not_frozen(cfg):
    fd = FrozenDetector(cfg)
    z = np.zeros_like(_ir(1))
    outs = [fd.update(_rgb(n), z, False, True) for n in range(1, 7)]
    assert not any(o["ir"]["frozen"] or o["ir"]["alarm"] for o in outs)
    assert outs[-1]["ir"]["diff_mad"] == 0.0


def test_missing_image_and_reset(cfg):
    fd = FrozenDetector(cfg)
    fd.update(_rgb(1), _ir(1), False, False)
    out = fd.update(None, _ir(1), True, False)
    assert out["rgb"] == {"frozen": False, "diff_mad": None, "alarm": False}
    fd.reset()
    assert fd.update(_rgb(1), _ir(1), False, False)["ir"]["diff_mad"] is None
