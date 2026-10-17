import math

import cv2
import numpy as np
import pytest

from baykus.health.features import (
    FEATURES,
    blur_score,
    frame_features,
    misalign,
    noise_sigma,
)
from baykus.robustness.corruptions import gaussian_blur, gaussian_noise, shift


def ramp(h: int = 512, w: int = 640) -> np.ndarray:
    """Düz eğim: Immerkær çekirdeği doğrusal eğimde sıfır yanıt verir."""
    return np.tile(np.linspace(30, 200, w), (h, 1)).astype(np.uint8)


def scene(seed: int = 0, h: int = 512, w: int = 640) -> np.ndarray:
    """Rastgele dikdörtgenlerden gri sahne (araç ve yol kenarı gibi kenar dokusu)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 90, np.uint8)
    for _ in range(80):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 30))
        bw, bh = int(rng.integers(8, 40)), int(rng.integers(6, 30))
        img[y : y + bh, x : x + bw] = int(rng.integers(0, 255))
    return img


def test_noise_sigma_flat_and_ramp_are_near_zero():
    assert noise_sigma(np.full((64, 64), 128, np.uint8)) == 0.0
    assert noise_sigma(ramp()) < 0.5


@pytest.mark.parametrize("sigma", [10, 20])
def test_noise_sigma_tracks_added_noise(sigma):
    noisy = gaussian_noise(ramp(), sigma, np.random.default_rng(0))
    assert noise_sigma(noisy) == pytest.approx(sigma, rel=0.1)


def test_noise_sigma_tiny_image_is_zero():
    assert noise_sigma(np.zeros((2, 2), np.uint8)) == 0.0


def test_blur_score_drops_with_blur_and_is_zero_on_flat():
    img = scene()
    assert blur_score(gaussian_blur(img, 7)) < blur_score(img)
    assert blur_score(np.zeros((32, 32), np.uint8)) == 0.0


def test_misalign_recovers_horizontal_shift_with_inverted_contrast():
    g = scene()
    bgr, ir = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 255 - g  # termal kutup tersi
    px, r = misalign(bgr, ir)
    assert px < 2.0 and r > 0
    px, _ = misalign(bgr, shift(ir, 20))
    assert px == pytest.approx(20, abs=2)


def test_misalign_with_lost_camera_is_none():
    g = scene()
    assert misalign(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), np.zeros_like(g)) == (None, None)
    assert misalign(np.zeros((512, 640, 3), np.uint8), g) == (None, None)


def test_frame_features_black_frames_have_no_nan():
    f = frame_features(np.zeros((512, 640, 3), np.uint8), np.zeros((512, 640), np.uint8))
    assert tuple(f) == FEATURES
    assert f["rgb_std"] == 0.0 and f["ir_std"] == 0.0 and f["misalign_px"] is None
    assert all(v is None or math.isfinite(v) for v in f.values())


def test_frame_features_values_are_plain_floats():
    g = scene()
    f = frame_features(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 255 - g)
    assert all(type(v) is float for v in f.values())


def test_misalign_is_deterministic():
    # float32 DFT çok iş parçacığında son basamakta oynuyordu: eşikler tekrarlanabilir olmalı
    g = scene(3)
    bgr, ir = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), shift(255 - g, 7)
    first = misalign(bgr, ir)
    assert all(misalign(bgr, ir) == first for _ in range(30))
