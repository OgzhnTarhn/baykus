import numpy as np
import pytest

from baykus.data.geometry import polygon_to_hbb, shift_and_clip, xyxy_to_yolo


def test_polygon_to_hbb_rotated_square():
    pts = np.array([[10, 0], [20, 10], [10, 20], [0, 10]], dtype=float)
    assert polygon_to_hbb(pts) == (0.0, 0.0, 20.0, 20.0)


def test_polygon_to_hbb_accepts_flat_list():
    assert polygon_to_hbb(np.array([1, 2, 5, 2, 5, 9, 1, 9])) == (1.0, 2.0, 5.0, 9.0)


def test_polygon_to_hbb_rejects_single_point():
    with pytest.raises(ValueError):
        polygon_to_hbb(np.array([[1.0, 2.0]]))


def test_shift_inside_crop():
    assert shift_and_clip((150, 120, 200, 160), 100, 100, 640, 512) == (50, 20, 100, 60)


def test_shift_partially_in_border_is_clipped():
    # sol üstte kenarlığa taşan kutu: x 80..130, y 90..140 → kırpma sonrası 0..30, 0..40
    assert shift_and_clip((80, 90, 130, 140), 100, 100, 640, 512) == (0, 0, 30, 40)


def test_shift_fully_in_border_is_dropped():
    assert shift_and_clip((10, 10, 60, 60), 100, 100, 640, 512) is None


def test_shift_fully_in_right_border_is_dropped():
    assert shift_and_clip((750, 300, 800, 340), 100, 100, 640, 512) is None


def test_degenerate_box_is_dropped():
    assert shift_and_clip((300, 300, 300.5, 320), 100, 100, 640, 512) is None


def test_xyxy_to_yolo():
    cx, cy, w, h = xyxy_to_yolo((0, 0, 64, 51.2), 640, 512)
    assert (cx, cy, w, h) == pytest.approx((0.05, 0.05, 0.1, 0.1))
