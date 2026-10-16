import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ensemble_boxes")

from baykus.data.illumination import mean_brightness  # noqa: E402
from baykus.fusion.late import (  # noqa: E402
    condition_levels,
    fuse_predictions,
    illumination_weights,
    levels_from_slices,
    load_brightness,
)
from baykus.robustness.corruptions import CLEAN, Condition, darken  # noqa: E402

IMAGES = {1: (640, 512), 2: (640, 512), 3: (640, 512)}
BOX = [100.0, 100.0, 20.0, 20.0]


def d(img, s, bbox=BOX, c=1):
    return {"image_id": img, "category_id": c, "bbox": list(bbox), "score": s}


def scores(dets):
    return {(x["image_id"], round(x["score"], 4)) for x in dets}


def test_fuse_per_image_and_empty_images():
    rgb = [d(1, 0.8)]
    ir = [d(1, 0.6), d(2, 0.8)]
    out = fuse_predictions(rgb, ir, IMAGES, (1.0, 1.0), iou_thr=0.55)
    assert scores(out) == {(1, 0.7), (2, 0.4)}  # 3. görüntü: iki model de boş → çıktı yok
    assert all(set(x) == {"image_id", "category_id", "bbox", "score"} for x in out)


def test_per_image_weights_k4():
    ir = [d(1, 0.8), d(2, 0.8)]
    w = {1: (1.0, 3.0), 2: (1.0, 1.0), 3: (1.0, 1.0)}
    assert scores(fuse_predictions([], ir, IMAGES, w, iou_thr=0.55)) == {(1, 0.6), (2, 0.4)}


def test_missing_weight_and_unknown_image():
    with pytest.raises(KeyError, match="ağırlık"):
        fuse_predictions([d(1, 0.8)], [], IMAGES, {2: (1.0, 1.0)}, iou_thr=0.55)
    with pytest.raises(ValueError, match="GT'de olmayan"):
        fuse_predictions([d(99, 0.8)], [], IMAGES, (1.0, 1.0), iou_thr=0.55)


def test_only_restricts_and_parallel_matches_serial():
    rgb = [d(i, 0.8) for i in IMAGES]
    ir = [d(i, 0.5) for i in IMAGES]
    assert {x["image_id"] for x in fuse_predictions(rgb, ir, IMAGES, (1.0, 2.0), iou_thr=0.55,
                                                    only=[2])} == {2}  # fmt: skip
    a = fuse_predictions(rgb, ir, IMAGES, (1.0, 2.0), iou_thr=0.55, workers=1)
    b = fuse_predictions(rgb, ir, IMAGES, (1.0, 2.0), iou_thr=0.55, workers=2)
    assert a == b


def test_levels_and_illumination_weights():
    levels = levels_from_slices({"day": [1], "night": [2, 3], "dark_night": []})
    assert levels == {1: "day", 2: "night", 3: "night"}
    with pytest.raises(ValueError, match="iki"):
        levels_from_slices({"day": [1], "night": [1]})
    w = illumination_weights(levels, {"day": 0.5, "night": 3.0})
    assert w == {1: (1.0, 0.5), 2: (1.0, 3.0), 3: (1.0, 3.0)}
    with pytest.raises(ValueError, match="night"):
        illumination_weights(levels, {"day": 0.5})


TH = (40.0, 114.14)  # (t_dark, t_day)


def test_condition_levels():
    b = {1: 200.0, 2: 60.0, 3: 20.0}
    assert condition_levels(b, TH, CLEAN) == {1: "day", 2: "night", 3: "dark_night"}
    assert condition_levels(b, TH, Condition("darken", 0.5, "rgb")) == {
        1: "night", 2: "dark_night", 3: "dark_night"}  # fmt: skip
    assert set(condition_levels(b, TH, Condition("drop_rgb", 1, "rgb")).values()) == {
        "dark_night"}  # fmt: skip
    for c in (Condition("noise", 5, "ir"), Condition("blur", 7, "both"),
              Condition("misalign", 10, "ir"), Condition("drop_ir", 1, "ir")):  # fmt: skip
        assert condition_levels(b, TH, c) == condition_levels(b, TH, CLEAN)


def test_darken_scales_brightness_within_half_level():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    base = mean_brightness(img)
    for f in (0.5, 0.25, 0.1):
        assert abs(mean_brightness(darken(img, f)) - f * base) <= 0.5


def test_load_brightness(tmp_path: Path):
    (tmp_path / "illumination.csv").write_text(
        "split,stem,brightness,illumination\nval,00101,150.25,day\nval,00103,30.00,dark_night\n"
        "train,00001,99.00,night\n",
        encoding="utf-8",
    )
    (tmp_path / "illumination_thresholds.json").write_text(
        json.dumps({"t_dark": 40.0, "t_day": 114.14}), encoding="utf-8"
    )
    imgs = [{"id": 1, "file_name": "00101"}, {"id": 2, "file_name": "00103"}]
    b, th = load_brightness(tmp_path, "val", imgs)
    assert b == {1: 150.25, 2: 30.0} and th == (40.0, 114.14)
    with pytest.raises(ValueError, match="00105"):
        load_brightness(tmp_path, "val", [*imgs, {"id": 3, "file_name": "00105"}])
