import numpy as np
import pytest

# (S) box_iou kullanıcı tarafından yazılır; dosya yokken bu testler atlanır, CI yeşil kalır.
box_iou = pytest.importorskip("baykus.eval.iou").box_iou


def test_identical():
    a = np.array([[0, 0, 10, 10]], float)
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)


def test_disjoint():
    assert box_iou(np.array([[0, 0, 1, 1]], float), np.array([[2, 2, 3, 3]], float))[0, 0] == 0


def test_half_overlap():
    # kesişim 5x10=50, birleşim 100+100-50=150
    r = box_iou(np.array([[0, 0, 10, 10]], float), np.array([[5, 0, 15, 10]], float))
    assert r[0, 0] == pytest.approx(50 / 150)


def test_shape_and_symmetry():
    a = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], float)
    b = np.array([[0, 0, 10, 10], [5, 5, 10, 10], [20, 20, 30, 30]], float)
    r = box_iou(a, b)
    assert r.shape == (2, 3)
    np.testing.assert_allclose(r, box_iou(b, a).T)


def test_zero_area_box_no_nan():
    r = box_iou(np.array([[1, 1, 1, 1]], float), np.array([[1, 1, 1, 1]], float))
    assert r[0, 0] == 0 and not np.isnan(r).any()


def test_empty_inputs():
    assert box_iou(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
