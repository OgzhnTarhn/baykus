import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.motion.images import half_gray, load_sequence

from .motion_synth import make_video, write_sequence

CLIP = Path(__file__).resolve().parents[2] / "fixtures" / "m3ot" / "clip" / "sequences" / "clip"


def test_half_gray_matches_health_edges_resize():
    ir = cv2.imread(str(CLIP / "ir" / "000001.png"), cv2.IMREAD_GRAYSCALE)
    h, w = ir.shape
    want = cv2.resize(ir, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    assert np.array_equal(half_gray(ir), want)


def test_half_gray_converts_bgr_first():
    bgr = cv2.imread(str(CLIP / "rgb" / "000001.jpg"), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    want = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    assert np.array_equal(half_gray(bgr), want)


def test_load_sequence_reads_lazily(tmp_path: Path):
    frames, gt = make_video(n=3)
    d = write_sequence(tmp_path, "s", frames, gt)
    seq = load_sequence(d, allow_test=False)
    assert seq.name == "s" and seq.split == "val" and seq.frames == [1, 2, 3]
    assert seq.ir(2).shape == frames[2].shape and seq.ir(2).ndim == 2
    assert seq.rgb(2).ndim == 3


def test_load_sequence_refuses_test(tmp_path: Path):
    frames, gt = make_video(n=2)
    d = write_sequence(tmp_path, "t", frames, gt, split="test")
    with pytest.raises(ValueError, match="test"):
        load_sequence(d, allow_test=False)
    assert load_sequence(d, allow_test=True).split == "test"
    assert json.loads((d / "index.json").read_text(encoding="utf-8"))["split"] == "test"
