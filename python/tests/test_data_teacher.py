"""Faz 4, Görev 15B: ``baykus.data.teacher`` saf fonksiyon testleri."""

from __future__ import annotations

from baykus.data.teacher import (
    CAR_CLS_ID,
    frame_vote,
    frames_for_track,
    track_class,
    window,
)


def test_window_centered() -> None:
    # 640 × 512 görüntünün ortasındaki kutu → merkezli 160 × 128 pencere.
    box = [300.0, 240.0, 340.0, 272.0]
    assert window(box, 640, 512) == (240, 192, 400, 320)


def test_window_shifted_at_border() -> None:
    # Sol üst köşedeki kutu → pencere (0, 0) ile başlar.
    assert window([0.0, 0.0, 20.0, 20.0], 640, 512) == (0, 0, 160, 128)
    # Sağ alt köşedeki kutu → pencere görüntü sınırına kaydırılır.
    assert window([620.0, 492.0, 640.0, 512.0], 640, 512) == (480, 384, 640, 512)


def test_frames_for_track_even_with_ends() -> None:
    frames = frames_for_track(list(range(1, 101)))
    assert len(frames) == 10
    assert frames[0] == 1
    assert frames[-1] == 100
    assert frames == sorted(frames)


def test_frames_for_track_short() -> None:
    assert frames_for_track([1, 2, 3]) == [1, 2, 3]


def test_frame_vote_best_iou_above_score() -> None:
    box = [0.0, 0.0, 10.0, 10.0]
    dets = [
        {"cls_id": 1, "score_raw": 0.9, "box": [0.0, 0.0, 8.0, 10.0]},  # IoU 0,8
        {"cls_id": 2, "score_raw": 0.6, "box": [0.0, 0.0, 10.0, 10.0]},  # IoU 1,0
    ]
    assert frame_vote(dets, box) == (2, 0.6)


def test_frame_vote_none_below_score() -> None:
    box = [0.0, 0.0, 10.0, 10.0]
    dets = [{"cls_id": 1, "score_raw": 0.4, "box": [0.0, 0.0, 10.0, 10.0]}]
    assert frame_vote(dets, box) is None


def test_frame_vote_none_below_iou() -> None:
    box = [0.0, 0.0, 10.0, 10.0]
    dets = [{"cls_id": 1, "score_raw": 0.9, "box": [100.0, 100.0, 110.0, 110.0]}]
    assert frame_vote(dets, box) is None


def test_track_class_majority() -> None:
    votes = [(1, 0.9), (1, 0.8), (2, 0.9)]
    assert track_class(votes) == 1


def test_track_class_tie_score_then_id() -> None:
    # Eşit sayıda oy → skor toplamı büyük olan kazanır.
    assert track_class([(1, 0.6), (2, 0.9)]) == 2
    # Skor toplamı da eşitse küçük cls_id kazanır.
    assert track_class([(2, 0.7), (1, 0.7)]) == 1


def test_track_class_empty_is_car() -> None:
    assert track_class([]) == CAR_CLS_ID
