import numpy as np
import pytest

from baykus.robustness.corruptions import (
    Condition,
    applies,
    apply_condition,
    darken,
    gaussian_blur,
    gaussian_noise,
    image_rng,
    shift,
    suite_conditions,
)


def test_darken_scales_and_rounds():
    img = np.array([[100, 255, 3]], np.uint8)
    out = darken(img, 0.5)
    assert out.dtype == np.uint8
    assert out.tolist() == [[50, 128, 2]]  # 127.5 → 128, 1.5 → 2 (np.rint)
    np.testing.assert_array_equal(darken(img, 1.0), img)


def test_darken_rejects_out_of_range():
    with pytest.raises(ValueError):
        darken(np.zeros((2, 2), np.uint8), 1.5)


def test_noise_zero_sigma_is_identity_and_seeded():
    img = np.full((64, 64), 128, np.uint8)
    np.testing.assert_array_equal(gaussian_noise(img, 0, np.random.default_rng(1)), img)
    a = gaussian_noise(img, 10, np.random.default_rng(1))
    b = gaussian_noise(img, 10, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert abs(float((a.astype(float) - 128).std()) - 10) < 0.5


def test_noise_clips_to_uint8():
    out = gaussian_noise(np.full((32, 32), 255, np.uint8), 20, np.random.default_rng(0))
    assert out.dtype == np.uint8 and out.max() == 255 and out.min() < 255


def test_blur_identity_constant_and_impulse():
    img = np.full((9, 9), 77, np.uint8)
    np.testing.assert_array_equal(gaussian_blur(img, 0), img)
    np.testing.assert_array_equal(gaussian_blur(img, 3), img)  # sabit görüntü değişmez
    imp = np.zeros((9, 9), np.uint8)
    imp[4, 4] = 255
    out = gaussian_blur(imp, 3)
    assert out[4, 4] < 255 and out[4, 5] > 0 and out[0, 0] == 0


def test_blur_rejects_even_kernel():
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((5, 5), np.uint8), 4)


def test_shift_right_left_and_out_of_frame():
    img = np.array([[1, 2, 3, 4]], np.uint8)
    assert shift(img, 1).tolist() == [[0, 1, 2, 3]]
    assert shift(img, -2).tolist() == [[3, 4, 0, 0]]
    assert shift(img, 4).tolist() == [[0, 0, 0, 0]]
    col = np.arange(3, dtype=np.uint8).reshape(3, 1)
    assert shift(col, 0, 1).ravel().tolist() == [0, 0, 1]
    rgb = np.ones((2, 3, 3), np.uint8)
    assert shift(rgb, 1).shape == (2, 3, 3) and shift(rgb, 1)[:, 0].sum() == 0


def test_suite_conditions_ids():
    ids = [c.id for c in suite_conditions()]
    assert ids == [
        "clean",
        "darken-0.5", "darken-0.25", "darken-0.1",
        "noise-5", "noise-10", "noise-20",
        "blur-3", "blur-7", "blur-11",
        "misalign-5", "misalign-10", "misalign-20", "misalign-40",
        "drop_rgb", "drop_ir",
    ]  # fmt: skip
    assert [c.id for c in suite_conditions(["darken"])] == [
        "clean", "darken-0.5", "darken-0.25", "darken-0.1"
    ]  # fmt: skip
    with pytest.raises(ValueError, match="gamma"):
        suite_conditions(["gamma"])


@pytest.mark.parametrize(
    "cid, rgb, ir, fusion",
    [
        ("clean", True, True, True),
        ("darken-0.5", True, False, True),
        ("noise-5", False, True, True),
        ("blur-3", True, True, True),
        ("misalign-5", False, False, True),
        ("drop_rgb", True, False, True),
        ("drop_ir", False, True, True),
    ],
)
def test_applies(cid, rgb, ir, fusion):
    cond = next(c for c in suite_conditions() if c.id == cid)
    assert (applies(cond, "rgb"), applies(cond, "ir"), applies(cond, "early_fusion")) == (
        rgb,
        ir,
        fusion,
    )


def test_image_rng_depends_on_stem_and_condition_only():
    c = Condition("noise", 10.0, "ir")
    a = image_rng(42, "00001", c).random(3)
    image_rng(42, "00002", c).random(3)  # araya başka görüntü girse de
    np.testing.assert_array_equal(a, image_rng(42, "00001", c).random(3))
    assert not np.array_equal(a, image_rng(42, "00002", c).random(3))
    assert not np.array_equal(a, image_rng(42, "00001", Condition("noise", 5.0, "ir")).random(3))


def test_apply_condition_targets_and_purity():
    bgr = np.full((8, 10, 3), 200, np.uint8)
    ir = np.full((8, 10), 100, np.uint8)
    b0, i0 = bgr.copy(), ir.copy()
    rb, ri = apply_condition(bgr, ir, Condition("darken", 0.5, "rgb"), seed=0, stem="x")
    assert rb.max() == 100 and np.array_equal(ri, ir)
    rb, ri = apply_condition(bgr, ir, Condition("misalign", 5.0, "ir"), seed=0, stem="x")
    assert np.array_equal(rb, bgr) and ri[:, :5].sum() == 0 and ri[:, 5:].min() == 100
    rb, ri = apply_condition(bgr, ir, Condition("drop_rgb", 1.0, "rgb"), seed=0, stem="x")
    assert rb.sum() == 0 and np.array_equal(ri, ir)
    rb, ri = apply_condition(bgr, ir, Condition("drop_ir", 1.0, "ir"), seed=0, stem="x")
    assert np.array_equal(rb, bgr) and ri.sum() == 0
    rb, ri = apply_condition(bgr, ir, Condition("blur", 3.0, "both"), seed=0, stem="x")
    assert rb.shape == bgr.shape and ri.shape == ir.shape
    np.testing.assert_array_equal(bgr, b0)  # girdiler değişmedi
    np.testing.assert_array_equal(ir, i0)


def test_late_fusion_sees_every_condition():
    for c in suite_conditions():
        assert applies(c, "late_fusion"), c.id
