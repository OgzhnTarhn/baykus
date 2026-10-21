import math

import numpy as np

from baykus.motion.geometry import apply, compose, corner_error, invert, scale, similarity, to_full


def _sim(tx: float, ty: float, deg: float, s: float) -> np.ndarray:
    c, k = s * math.cos(math.radians(deg)), s * math.sin(math.radians(deg))
    return np.array([[c, -k, tx], [k, c, ty]], np.float64)


def test_to_full_maps_half_pixel_centers():
    # Yarı çözünürlük piksel i ↔ tam 2i + 0,5: tam p → yarı (p − 0,5)/2 → A → tam 2q + 0,5.
    a = _sim(3.0, -2.0, 0.7, 1.02)
    p = np.array([[0.0, 0.0], [639.0, 511.0], [100.5, 37.25]])
    q = apply(a, (p - 0.5) / 2) * 2 + 0.5
    assert np.allclose(apply(to_full(a), p), q)


def test_to_full_translation_doubles():
    assert np.allclose(to_full(_sim(1.0, -3.0, 0.0, 1.0)), [[1, 0, 2], [0, 1, -6]])


def test_invert_and_compose_round_trip():
    a, b = _sim(5, 7, 1.0, 0.97), _sim(-3, 2, -0.5, 1.03)
    assert np.allclose(compose(a, invert(a)), [[1, 0, 0], [0, 1, 0]])
    p = np.array([[10.0, 20.0], [300.0, 4.0]])
    assert np.allclose(apply(compose(a, b), p), apply(a, apply(b, p)))


def test_scale_is_sqrt_abs_det():
    assert math.isclose(scale(_sim(9, 9, 13.0, 2.0)), 2.0)
    assert math.isclose(scale(np.array([[1.0, 0, 0], [0, -4.0, 0]])), 2.0)


def test_corner_error_is_max_corner_distance():
    i = _sim(0, 0, 0, 1)
    assert corner_error(i, i, 640, 512) == 0.0
    assert math.isclose(corner_error(i, _sim(3, 4, 0, 1), 640, 512), 5.0)
    # Ölçek farkı en uzak köşede en büyük: (640, 512) · 0,01.
    assert math.isclose(corner_error(i, _sim(0, 0, 0, 1.01), 640, 512), math.hypot(6.4, 5.12))


def test_similarity_about_center_keeps_center():
    m = similarity(0.0, 0.0, 1.0, 1.03, center=(320.0, 256.0))
    assert np.allclose(apply(m, np.array([[320.0, 256.0]])), [[320.0, 256.0]])
    assert math.isclose(scale(m), 1.03)
    assert np.allclose(similarity(5.0, -2.0, 0.0, 1.0), [[1, 0, 5], [0, 1, -2]])
