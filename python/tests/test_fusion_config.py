from pathlib import Path

import pytest

from baykus.fusion.config import load_fusion_config, parse_fusion_config
from baykus.train.config import ConfigError

BASE = {
    "experiment": "e4",
    "mode": "late_fusion",
    "model": "yolo26s.pt",
    "rgb": "configs/e1_rgb.yaml",
    "ir": "configs/e2_ir.yaml",
    "wbf": {"iou_thr": [0.5, 0.6], "ir_weight": [1, 2]},
}


def test_parse_defaults_and_hash():
    c = parse_fusion_config(BASE)
    assert c.iou_thr == (0.5, 0.6) and c.ir_weight == (1.0, 2.0)
    assert c.skip_box_thr == 0.001 and c.per_illumination is False
    assert c.label_sets == ("ir", "rgb") and c.n_boot == 1000 and c.seed == 42
    assert c.config_hash.startswith("sha256:")


@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"mode": "rgb"}, "late_fusion"),
        ({"foo": 1}, "foo"),
        ({"wbf": {"iou_thr": [0.5], "ir_weight": []}}, "ir_weight"),
        ({"wbf": {"iou_thr": [0.5], "ir_weight": [0]}}, "ir_weight"),
        ({"wbf": {"iou_thr": [1.2], "ir_weight": [1]}}, "iou_thr"),
        ({"wbf": {"ir_weight": [1]}}, "iou_thr"),
        ({"wbf": {"iou_thr": [0.5], "ir_weight": [1], "per_illumination": True}}, "E4"),
        ({"eval": {"label_sets": ["x"]}}, "label_sets"),
    ],
)
def test_parse_errors(patch, msg):
    with pytest.raises(ConfigError, match=msg):
        parse_fusion_config({**BASE, **patch})


def test_repo_fusion_configs():
    root = Path(__file__).resolve().parents[2] / "configs" / "fusion"
    e4 = load_fusion_config(root / "e4_late_fusion.yaml")
    e4b = load_fusion_config(root / "e4b_late_fusion_illum.yaml")
    assert e4.per_illumination is False and len(e4.iou_thr) * len(e4.ir_weight) == 12
    assert e4b.per_illumination is True and e4b.iou_thr == () and e4b.ir_weight == e4.ir_weight
