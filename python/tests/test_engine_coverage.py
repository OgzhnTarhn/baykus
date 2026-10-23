"""engine_ref.coverage: koşul uygulama, açgözlü bulma oranı ve CLI test bölümü reddi."""

from __future__ import annotations

import json

import numpy as np
import pytest

from baykus.engine_ref.cli import main
from baykus.engine_ref.coverage import apply, recall


def test_coverage_rgb_off_zeroes_rgb() -> None:
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    ir = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    b, i = apply("rgb_off", bgr, ir)
    assert not b.any()
    assert np.array_equal(i, ir)


def test_coverage_ir_as_rgb_copies_ir() -> None:
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    ir = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    b, i = apply("ir_as_rgb", bgr, ir)
    for c in range(3):
        assert np.array_equal(b[:, :, c], ir)
    assert np.array_equal(i, ir)


def test_coverage_unknown_condition_raises() -> None:
    bgr = np.zeros((4, 4, 3), np.uint8)
    ir = np.zeros((4, 4), np.uint8)
    with pytest.raises(ValueError):
        apply("nope", bgr, ir)


def test_recall_greedy_one_to_one() -> None:
    gt = [[0.0, 0.0, 10.0, 10.0]]
    dets = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
    assert recall(dets, gt, 0.5) == 1


def test_recall_iou_threshold() -> None:
    gt = [[0.0, 0.0, 10.0, 10.0]]
    dets = [[0.0, 0.0, 10.0, 4.0]]  # IoU = 40 / 100 = 0,4
    assert recall(dets, gt, 0.5) == 0
    assert recall(dets, gt, 0.3) == 1


def test_coverage_cli_rejects_test_split(tmp_path, capsys) -> None:
    seq = tmp_path / "seq"
    seq.mkdir()
    (seq / "index.json").write_text(
        json.dumps({"name": "s", "split": "test", "frames": []}), encoding="utf-8"
    )
    # modeller "yok": test denetimi modellerden önce çalışmazsa hata "yok" hakkında olurdu
    rc = main(
        ["coverage", "--models", "yok", "--seq-dir", str(seq), "--labels", "yok", "--out", "yok"]
    )
    assert rc == 2
    assert "test" in capsys.readouterr().err
