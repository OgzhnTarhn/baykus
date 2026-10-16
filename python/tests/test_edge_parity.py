"""İki tahmin listesinin eşliği (H5 Görev 10; FR-E6 toleransları)."""

from baykus.edge.parity import compare_dets

B = {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}


def test_identical_ok():
    r = compare_dets([B], [B])
    assert r["ok"] is True and r["matched"] == 1 and r["min_iou"] == 1.0


def test_shift_breaks_iou():
    assert compare_dets([B], [{**B, "bbox": [11, 10, 20, 20]}])["ok"] is False


def test_missing_box():
    r = compare_dets([B, {**B, "image_id": 2}], [B])
    assert (r["ok"], r["matched"], r["count_a"], r["count_b"]) == (False, 1, 2, 1)


def test_score_tolerance():
    assert compare_dets([B], [{**B, "score": 0.902}])["ok"] is False
    assert compare_dets([B], [{**B, "score": 0.9005}])["ok"] is True


def test_low_scores_ignored():
    assert compare_dets([B, {**B, "score": 0.1}], [B])["ok"] is True


def test_class_must_match():
    assert compare_dets([B], [{**B, "category_id": 2}])["ok"] is False


def test_both_empty_ok():
    r = compare_dets([], [])
    assert r["ok"] is True and r["min_iou"] is None and r["max_score_diff"] is None
