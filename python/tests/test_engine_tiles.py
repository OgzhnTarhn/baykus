"""`baykus.engine_ref.tiles` birim testleri: pencere geometrisi ve sınıfsız NMS birleştirme."""

from __future__ import annotations

import numpy as np
import pytest

from baykus.engine_ref.pipeline import Det, Pipeline
from baykus.engine_ref.tiles import merge, windows


def _det(index: int, cls_id: int, score: float, box: tuple[float, float, float, float]) -> Det:
    return Det(index, cls_id, score, score, box)


def test_windows_640x512() -> None:
    assert windows(640, 512) == [
        (0, 0, 360, 288),
        (280, 0, 360, 288),
        (0, 224, 360, 288),
        (280, 224, 360, 288),
    ]


def test_windows_scale_other_size() -> None:
    wins = windows(800, 600)
    assert [(w[2], w[3]) for w in wins] == [(450, 338)] * 4
    assert [w[0] for w in wins] == [0, 350, 0, 350]
    assert [w[1] for w in wins] == [0, 0, 262, 262]


def test_windows_tiny_frame() -> None:
    wins = windows(33, 17)
    assert [(w[2], w[3]) for w in wins] == [(19, 10)] * 4
    assert [w[0] for w in wins] == [0, 14, 0, 14]
    assert [w[1] for w in wins] == [0, 0, 7, 7]
    for x, y, pw, ph in wins:
        assert pw > 0 and ph > 0
        assert x >= 0 and y >= 0
        assert x + pw <= 33 and y + ph <= 17


def test_merge_shifts_boxes() -> None:
    wins = windows(640, 512)
    per_tile = [[], [_det(0, 0, 0.9, (0.0, 0.0, 10.0, 10.0))], [], []]
    out = merge(per_tile, wins)
    assert len(out) == 1
    assert out[0].box == (280.0, 0.0, 290.0, 10.0)


def test_merge_nms_class_agnostic() -> None:
    wins = windows(640, 512)
    per_tile = [
        [_det(0, 0, 0.9, (280.0, 0.0, 290.0, 10.0))],
        [_det(0, 1, 0.8, (0.0, 0.0, 10.0, 10.0))],
        [],
        [],
    ]
    out = merge(per_tile, wins)
    assert len(out) == 1
    assert out[0].cls_id == 0
    assert out[0].score_raw == 0.9


def test_merge_keeps_low_iou() -> None:
    wins = windows(640, 512)
    per_tile = [
        [
            _det(0, 0, 0.9, (0.0, 0.0, 10.0, 10.0)),
            _det(1, 0, 0.8, (5.0, 0.0, 15.0, 10.0)),
        ],
        [],
        [],
        [],
    ]
    out = merge(per_tile, wins)
    assert len(out) == 2


def test_merge_order_ties() -> None:
    wins = windows(640, 512)
    per_tile = [
        [_det(0, 0, 0.9, (0.0, 0.0, 10.0, 10.0))],
        [_det(0, 1, 0.9, (0.0, 0.0, 10.0, 10.0))],
        [],
        [],
    ]
    out = merge(per_tile, wins)
    assert len(out) == 2
    assert out[0].cls_id == 0
    assert out[0].index == 0


def test_merge_reindexes() -> None:
    wins = windows(640, 512)
    per_tile = [
        [_det(0, 0, 0.9, (0.0, 0.0, 10.0, 10.0))],
        [_det(0, 1, 0.8, (0.0, 0.0, 10.0, 10.0))],
        [_det(0, 2, 0.7, (0.0, 0.0, 10.0, 10.0))],
        [_det(0, 3, 0.6, (0.0, 0.0, 10.0, 10.0))],
    ]
    out = merge(per_tile, wins)
    assert sorted(d.index for d in out) == list(range(len(out)))


class _FakeBackend:
    imgsz = 640

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.array([[[250.0, 250.0, 350.0, 350.0, 0.9, 0.0]]], dtype=np.float32)


class _Models:
    calib_x = np.array([0.0, 1.0])
    calib_y = np.array([0.0, 1.0])
    thresholds: dict = {}
    gate = None


def test_detect_tiles4_merges_four_crops() -> None:
    backend = _FakeBackend()
    pipe = Pipeline(_Models(), backend)
    bgr = np.zeros((512, 640, 3), dtype=np.uint8)
    ir = np.zeros((512, 640), dtype=np.uint8)

    out = pipe.detect(bgr, ir, tiles=4)
    assert backend.calls == 4
    assert out["windows"] == windows(640, 512)
    assert out["dets_all"]
    for d in out["dets_all"]:
        x1, y1, x2, y2 = d.box
        assert 0.0 <= x1 < x2 <= 640.0
        assert 0.0 <= y1 < y2 <= 512.0

    backend.calls = 0
    pipe.detect(bgr, ir, tiles=1)
    assert backend.calls == 1

    with pytest.raises(ValueError):
        pipe.detect(bgr, ir, tiles=2)
