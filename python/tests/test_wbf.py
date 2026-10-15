"""WBF sarmalayıcısı (S): elle hesaplanmış değerler (kural: H4 Görev 4, ensemble-boxes 1.0.9)."""

import warnings

import pytest

pytest.importorskip("ensemble_boxes")

from baykus.fusion.wbf import fuse_image  # noqa: E402

W, H = 100, 50


def det(x, y, w, h, s, c=1):
    return {"category_id": c, "bbox": [x, y, w, h], "score": s}


def fuse(a, b, weights=(1, 1), iou_thr=0.55, skip=0.0):
    return fuse_image([a, b], list(weights), width=W, height=H, iou_thr=iou_thr, skip_box_thr=skip)


def test_identical_boxes_equal_weights_average_score():
    out = fuse([det(10, 10, 20, 10, 0.8)], [det(10, 10, 20, 10, 0.6)])
    assert len(out) == 1
    assert out[0]["score"] == pytest.approx(0.7, abs=1e-5)  # (0.8 + 0.6) / 2 · 2/2
    assert out[0]["bbox"] == pytest.approx([10, 10, 20, 10], abs=1e-3)
    assert out[0]["category_id"] == 1


def test_box_from_one_model_is_scaled_by_its_weight_share():
    assert fuse([det(10, 10, 20, 10, 0.8)], [])[0]["score"] == pytest.approx(0.4, abs=1e-5)
    # K4: ağırlık oranı skoru belirler: 0.8 · 2/3 ve 0.6 · 1/3
    one = fuse([det(10, 10, 20, 10, 0.8)], [], (2, 1))
    other = fuse([], [det(10, 10, 20, 10, 0.6)], (2, 1))
    assert one[0]["score"] == pytest.approx(0.5333, abs=1e-4)
    assert other[0]["score"] == pytest.approx(0.2, abs=1e-5)


def test_weighted_coordinates_and_score():
    # A = [0,0,10,10], B = [1,1,11,11] (x1y1x2y2); IoU 81/119 = 0.68 > 0.55 → tek küme.
    # Ağırlıklı skorlar 1.6 ve 0.6 → x1 = 0.6·1 / 2.2 = 0.2727; skor = 1.1 · 2/3 = 0.7333
    out = fuse([det(0, 0, 10, 10, 0.8)], [det(1, 1, 10, 10, 0.6)], (2, 1))
    assert len(out) == 1
    assert out[0]["bbox"] == pytest.approx([0.2727, 0.2727, 10.0, 10.0], abs=1e-3)
    assert out[0]["score"] == pytest.approx(0.7333, abs=1e-4)


def test_low_iou_stays_separate_and_output_is_sorted():
    # A = [0,0,10,10], B = [5,0,15,10]: IoU 50/150 = 0.33 < 0.55
    out = fuse([det(0, 0, 10, 10, 0.6)], [det(5, 0, 10, 10, 0.8)])
    assert [round(d["score"], 4) for d in out] == [0.4, 0.3]
    assert out[0]["bbox"] == pytest.approx([5, 0, 10, 10], abs=1e-3)


def test_classes_never_merge():
    out = fuse([det(10, 10, 20, 10, 0.8, c=1)], [det(10, 10, 20, 10, 0.6, c=2)])
    assert sorted((d["category_id"], round(d["score"], 4)) for d in out) == [(1, 0.4), (2, 0.3)]


def test_outside_box_is_clipped_and_empty_box_dropped_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # kırpma bizde: kütüphane uyarı basmamalı
        out = fuse([det(-5, 0, 10, 10, 0.8), det(-30, 0, 10, 10, 0.9)], [])
    assert len(out) == 1
    assert out[0]["bbox"] == pytest.approx([0, 0, 5, 10], abs=1e-3)


def test_skip_threshold_and_zero_score():
    assert fuse([det(10, 10, 20, 10, 0.0005)], [], skip=0.001) == []
    assert fuse([det(10, 10, 20, 10, 0.0)], []) == []  # 0 skor: 0'a bölme / NaN yok


def test_no_boxes():
    assert fuse([], []) == []


def test_bad_weights():
    with pytest.raises(ValueError, match="ağırlık"):
        fuse_image([[], []], [1.0], width=W, height=H, iou_thr=0.5)
    with pytest.raises(ValueError, match="ağırlık"):
        fuse([det(10, 10, 20, 10, 0.8)], [], (1, 0))


def test_types_and_input_untouched():
    a = [det(10, 10, 20, 10, 0.8)]
    before = [dict(d, bbox=list(d["bbox"])) for d in a]
    out = fuse(a, [])
    assert a == before
    d = out[0]
    assert set(d) == {"category_id", "bbox", "score"}
    assert type(d["category_id"]) is int and type(d["score"]) is float
    assert all(type(v) is float for v in d["bbox"])
