import pytest

pytest.importorskip("ensemble_boxes")

from baykus.fusion.config import parse_fusion_config  # noqa: E402
from baykus.fusion.select import (  # noqa: E402
    best,
    grid_search,
    select_constant,
    select_per_illumination,
)
from tests.coco_helpers import make_gt  # noqa: E402

A, B = [100.0, 100.0, 20.0, 20.0], [300.0, 300.0, 20.0, 20.0]
GT = make_gt({1: [(1, A)], 2: [(1, A)]})  # 1 gündüz, 2 gece; araç A'da


def p(img, bbox, s):
    return {"image_id": img, "category_id": 1, "bbox": list(bbox), "score": s}


# Gündüz: RGB A'yı 0.5 ile buluyor, IR B'de 0.9'luk yanlış pozitif. Gece: tersi.
# Tek modelden gelen kutunun skoru s·w / (w_rgb + w_ir) → ir_weight 0.5 gündüz, 3 gece kazanır.
RGB = [p(1, A, 0.5), p(2, B, 0.9)]
IR = [p(1, B, 0.9), p(2, A, 0.5)]
CFG = parse_fusion_config({
    "experiment": "e4", "mode": "late_fusion", "model": "yolo26s.pt", "rgb": "r", "ir": "i",
    "wbf": {"iou_thr": [0.5, 0.7], "ir_weight": [0.5, 3.0]},
})  # fmt: skip


def test_grid_search_scores_each_cell():
    rows = grid_search(RGB, IR, GT, iou_thrs=[0.5], ir_weights=[0.5, 3.0], skip_box_thr=0.001,
                       img_ids=[1])  # fmt: skip
    assert [(r["ir_weight"], round(r["map50"], 3)) for r in rows] == [(0.5, 1.0), (3.0, 0.5)]


def test_best_takes_first_on_tie():
    rows = [{"iou_thr": 0.5, "ir_weight": 1.0, "map50": 0.8},
            {"iou_thr": 0.6, "ir_weight": 2.0, "map50": 0.8}]  # fmt: skip
    assert best(rows) is rows[0]
    with pytest.raises(ValueError):
        best([])


def test_select_constant_tie_breaks_to_grid_order():
    # İki görüntü birlikte: her ağırlıkta bir FP bir TP'nin önünde → hepsi 0.5, ilk hücre seçilir
    out = select_constant(RGB, IR, GT, CFG)
    assert len(out["grid"]) == 4
    assert out["selected"] == {"iou_thr": 0.5, "w_rgb": 1.0, "ir_weight": 0.5}


def test_select_per_illumination_picks_per_slice():
    slices = {"day": [1], "night": [2], "dark_night": []}
    out = select_per_illumination(RGB, IR, GT, CFG, slices, iou_thr=0.6)
    assert out["selected"] == {"iou_thr": 0.6, "w_rgb": 1.0,
                               "ir_weight": {"day": 0.5, "night": 3.0}}  # fmt: skip
    assert {r["slice"] for r in out["grid"]} == {"day", "night"}  # boş dilim atlanır
