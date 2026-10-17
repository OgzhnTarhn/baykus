import json
from pathlib import Path

import pytest

pytest.importorskip("ensemble_boxes")

from baykus.eval.predictions import load_predictions, save_predictions  # noqa: E402
from baykus.fusion.cli import main, selection_path  # noqa: E402
from tests.coco_helpers import make_gt  # noqa: E402
from tests.test_fusion_select import IR, RGB  # noqa: E402

A = [100.0, 100.0, 20.0, 20.0]
REPO = Path(__file__).resolve().parents[2]


def setup_dv(tmp_path: Path) -> dict:
    """İki görüntülü val: 1 gündüz, 2 gece; GT her iki etiket setinde aynı."""
    root, meta = tmp_path / "dv", tmp_path / "meta"
    (root / "coco").mkdir(parents=True)
    meta.mkdir()
    gt = make_gt({1: [(1, A)], 2: [(1, A)]})
    for im, stem in zip(gt.dataset["images"], ("00101", "00103"), strict=True):
        im["file_name"] = stem
    for ls in ("ir", "rgb"):
        (root / "coco" / f"val_{ls}.json").write_text(json.dumps(gt.dataset), encoding="utf-8")
    slices = {"illumination": {"day": [1], "night": [2], "dark_night": []}}
    (meta / "slices_val.json").write_text(json.dumps(slices), encoding="utf-8")
    (meta / "illumination.csv").write_text(
        "split,stem,brightness,illumination\nval,00101,150.00,day\nval,00103,60.00,night\n",
        encoding="utf-8",
    )
    (meta / "illumination_thresholds.json").write_text(
        json.dumps({"t_dark": 40.0, "t_day": 114.14}), encoding="utf-8"
    )
    # değerlendirme klasörü adı run kimliğiyle başlar (check_sources)
    rgb = save_predictions(tmp_path / "E1_val" / "predictions_coco.json.gz", RGB)
    ir = save_predictions(tmp_path / "E2_val" / "predictions_coco.json.gz", IR)
    return {"root": root, "meta": meta, "rgb": rgb, "ir": ir}


def select_args(tmp_path: Path, s: dict, config: str, extra=()) -> list[str]:
    return ["select", "--config", str(REPO / "configs/fusion" / config),
            "--rgb-preds", str(s["rgb"]), "--ir-preds", str(s["ir"]),
            "--rgb-run-id", "E1", "--ir-run-id", "E2", "--data-root", str(s["root"]),
            "--meta", str(s["meta"]), "--project", str(tmp_path / "eval"),
            "--results", str(tmp_path / "res"), "--n-boot", "10", *extra]  # fmt: skip


def test_select_e4_writes_selection_runs_and_predictions(tmp_path: Path):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    sel = json.loads(selection_path(tmp_path / "res", "e4_late_fusion", "val").read_text(
        encoding="utf-8"))  # fmt: skip
    assert len(sel["grid"]) == 12 and sel["criterion"] == "map50"
    assert sel["selected"]["w_rgb"] == 1.0 and sel["rgb_run_id"] == "E1"
    run_id = sel["run_id"]
    assert "_e4-late-fusion_s_" in run_id
    for ls in ("ir", "rgb"):
        rec = json.loads((tmp_path / "res/runs" / f"{run_id}_val_{ls}.json").read_text(
            encoding="utf-8"))  # fmt: skip
        assert rec["model"]["mode"] == "late_fusion" and rec["weights"] is None
        assert rec["fusion"]["ir_weight"] == sel["selected"]["ir_weight"]
        assert rec["metrics"]["per_illumination"]["day"] is not None
    assert load_predictions(tmp_path / "eval" / f"{run_id}_val" / "predictions_coco.json.gz")


def test_select_resumes_without_searching_again(tmp_path: Path, monkeypatch, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    import baykus.fusion.cli as fc

    def boom(*a, **kw):
        raise AssertionError("ızgara yeniden arandı")

    monkeypatch.setattr(fc, "select_constant", boom)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    assert "zaten var" in capsys.readouterr().out


def test_select_resume_with_other_models_is_error(tmp_path: Path, capsys):
    # E1 yeniden eğitilirse eski seçim yeni tahminlere sessizce uygulanmamalı
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    args = select_args(tmp_path, s, "e4_late_fusion.yaml")
    args[args.index("E1")] = "E1x"  # yeni modelin tahminleri kendi klasöründe
    new = save_predictions(tmp_path / "E1x_val" / "predictions_coco.json.gz", RGB)
    args[args.index(str(s["rgb"]))] = str(new)
    assert main(args) == 2 and "başka modeller" in capsys.readouterr().err


def test_select_e4b_needs_base_and_picks_per_slice(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4b_late_fusion_illum.yaml")) == 2
    assert "--base" in capsys.readouterr().err
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    base = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    assert main(select_args(tmp_path, s, "e4b_late_fusion_illum.yaml",
                            ["--base", str(base)])) == 0  # fmt: skip
    sel = json.loads(selection_path(tmp_path / "res", "e4b_late_fusion_illum", "val").read_text(
        encoding="utf-8"))  # fmt: skip
    # repo ızgarası [0.5, 1, 2, 3]: gecede 2 ve 3 eşit (AP 1.0) → eşitlikte ilk: 2.0
    assert sel["selected"]["ir_weight"] == {"day": 0.5, "night": 2.0}
    assert sel["selected"]["iou_thr"] == json.loads(base.read_text(encoding="utf-8"))[
        "selected"]["iou_thr"]  # fmt: skip
    rec = json.loads((tmp_path / "res/runs" / f"{sel['run_id']}_val_ir.json").read_text(
        encoding="utf-8"))  # fmt: skip
    assert rec["metrics"]["map50"] == pytest.approx(1.0)  # her karede TP önde


def test_select_base_from_other_models_is_error(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    base = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    args = select_args(tmp_path, s, "e4b_late_fusion_illum.yaml", ["--base", str(base)])
    args[args.index("E2")] = "E2x"
    new = save_predictions(tmp_path / "E2x_val" / "predictions_coco.json.gz", IR)
    args[args.index(str(s["ir"]))] = str(new)
    assert main(args) == 2 and "başka modeller" in capsys.readouterr().err


from baykus.fusion.cli import compose_sources  # noqa: E402
from baykus.robustness.corruptions import CLEAN, Condition  # noqa: E402


@pytest.mark.parametrize(
    ("cond", "want"),
    [
        (CLEAN, ("clean", "clean")),
        (Condition("darken", 0.5, "rgb"), ("darken-0.5", "clean")),
        (Condition("noise", 5, "ir"), ("clean", "noise-5")),
        (Condition("blur", 3, "both"), ("blur-3", "blur-3")),
        (Condition("misalign", 10, "ir"), ("clean", "misalign-10")),
        (Condition("drop_rgb", 1, "rgb"), ("drop_rgb", "clean")),
        (Condition("drop_ir", 1, "ir"), ("clean", "drop_ir")),
    ],
)
def test_compose_sources(cond, want):
    assert compose_sources(cond) == want


def _robust_preds(tmp_path: Path, conds: dict[str, list[str]]) -> Path:
    d = tmp_path / "robust"
    for run, ids in conds.items():
        for cid in ids:
            dets = RGB if run == "E1" else IR
            save_predictions(d / f"{run}_val_{cid}" / "predictions_coco.json.gz", dets)
    return d


def _robust_args(tmp_path, s, rp, extra=()):
    sel = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    return ["robust", "--config", str(REPO / "configs/fusion/e4_late_fusion.yaml"),
            "--selection", str(sel), "--robust-preds", str(rp), "--data-root", str(s["root"]),
            "--meta", str(s["meta"]), "--project", str(tmp_path / "rob"),
            "--results", str(tmp_path / "res"), "--n-boot", "10", *extra]  # fmt: skip


def test_robust_composes_condition_predictions(tmp_path: Path):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    rp = _robust_preds(tmp_path, {"E1": ["clean", "drop_rgb"], "E2": ["clean"]})
    assert main(_robust_args(tmp_path, s, rp, ["--only", "drop_rgb"])) == 0
    files = sorted(p.name.split("_val_")[1] for p in (tmp_path / "res/robust").glob("*.json"))
    assert files == ["clean_ir.json", "drop_rgb_ir.json"]
    rec = json.loads(next((tmp_path / "res/robust").glob("*drop_rgb_ir.json")).read_text(
        encoding="utf-8"))  # fmt: skip
    assert rec["robustness"]["sources"] == {"rgb": "drop_rgb", "ir": "clean"}
    assert rec["model"]["mode"] == "late_fusion" and rec["label_set"] == "ir"
    assert list((tmp_path / "rob").glob("*_val_drop_rgb/predictions_coco.json.gz"))
    assert main(_robust_args(tmp_path, s, rp, ["--only", "drop_rgb"])) == 0  # tekrar: atlanır


def test_robust_missing_inputs_fail_before_any_work(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    rp = _robust_preds(tmp_path, {"E1": ["clean"], "E2": ["clean"]})
    assert main(_robust_args(tmp_path, s, rp, ["--only", "misalign"])) == 2
    err = capsys.readouterr().err
    assert "E2_val_misalign-5" in err and "--ir-misalign" in err
    assert not (tmp_path / "res/robust").exists()


def _m3ot_preds(tmp_path: Path, rgb=RGB, ir=IR) -> tuple[Path, Path]:
    """M3OT değerlendirme klasörleri (baykus.eval.cli.run_name: <run>_m3ot_<split>)."""
    name = "predictions_coco.json.gz"
    return (save_predictions(tmp_path / "E1_m3ot_val" / name, rgb),
            save_predictions(tmp_path / "E2_m3ot_val" / name, ir))  # fmt: skip


def test_apply_on_other_dataset_and_e4b_refused(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    sel = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    m_rgb, m_ir = _m3ot_preds(tmp_path)
    args = ["apply", "--config", str(REPO / "configs/fusion/e4_late_fusion.yaml"),
            "--selection", str(sel), "--rgb-preds", str(m_rgb), "--ir-preds", str(m_ir),
            "--data-root", str(s["root"]), "--meta", str(s["meta"]), "--split", "val",
            "--dataset", "m3ot", "--class-agnostic", "--project", str(tmp_path / "m"),
            "--results", str(tmp_path / "res"), "--n-boot", "10"]  # fmt: skip
    assert main(args) == 0
    run_id = json.loads(sel.read_text(encoding="utf-8"))["run_id"]
    rec = json.loads((tmp_path / "res/m3ot" / f"{run_id}_m3ot_val_ir.json").read_text(
        encoding="utf-8"))  # fmt: skip
    assert rec["dataset"] == "m3ot" and rec["class_agnostic"] is True
    base = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    assert main(select_args(tmp_path, s, "e4b_late_fusion_illum.yaml",
                            ["--base", str(base)])) == 0  # fmt: skip
    e4b = selection_path(tmp_path / "res", "e4b_late_fusion_illum", "val")
    args[args.index(str(sel))] = str(e4b)
    args[args.index(str(REPO / "configs/fusion/e4_late_fusion.yaml"))] = str(
        REPO / "configs/fusion/e4b_late_fusion_illum.yaml")  # fmt: skip
    assert main(args) == 2 and "uygulanmaz" in capsys.readouterr().err


def test_rerun_with_other_n_boot_is_error(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    args = select_args(tmp_path, s, "e4_late_fusion.yaml")
    args[args.index("--n-boot") + 1] = "20"
    assert main(args) == 2 and "farklı girdilerle" in capsys.readouterr().err


def test_rerun_with_other_sources_is_error(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    other = save_predictions(tmp_path / "x" / "E1_val" / "predictions_coco.json.gz", RGB)
    args = select_args(tmp_path, s, "e4_late_fusion.yaml")
    args[args.index(str(s["rgb"]))] = str(other)
    # seçim başka tahminlerle yapılmış: aramadan önce yakalanır (birleştirmeye gelmez)
    assert main(args) == 2 and "tahminlerle" in capsys.readouterr().err


def _apply_args(tmp_path: Path, s: dict, config: Path, rgb: Path, ir: Path) -> list[str]:
    sel = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    return ["apply", "--config", str(config), "--selection", str(sel), "--rgb-preds", str(rgb),
            "--ir-preds", str(ir), "--data-root", str(s["root"]), "--meta", str(s["meta"]),
            "--split", "val", "--dataset", "m3ot", "--class-agnostic",
            "--project", str(tmp_path / "m"), "--results", str(tmp_path / "res"),
            "--n-boot", "10"]  # fmt: skip


def test_apply_preds_from_other_run_is_error(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    wrong = save_predictions(tmp_path / "E1x_val" / "predictions_coco.json.gz", RGB)
    cfg = REPO / "configs/fusion/e4_late_fusion.yaml"
    assert main(_apply_args(tmp_path, s, cfg, wrong, s["ir"])) == 2
    assert "E1" in capsys.readouterr().err


def test_apply_config_hash_mismatch_is_error(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    src = (REPO / "configs/fusion/e4_late_fusion.yaml").read_text(encoding="utf-8")
    cfg = tmp_path / "e4_changed.yaml"
    cfg.write_text(src.replace("n_boot: 1000", "n_boot: 999"), encoding="utf-8")
    assert main(_apply_args(tmp_path, s, cfg, s["rgb"], s["ir"])) == 2
    assert "config_hash" in capsys.readouterr().err


def test_value_error_becomes_error_line(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    rgb, ir = _m3ot_preds(tmp_path, rgb=[{**RGB[0], "image_id": 99}])
    cfg = REPO / "configs/fusion/e4_late_fusion.yaml"
    assert main(_apply_args(tmp_path, s, cfg, rgb, ir)) == 2
    err = capsys.readouterr().err
    assert err.startswith("hata: ") and "Traceback" not in err


def _dv_apply_args(tmp_path: Path, s: dict, sel: Path, config: str, rgb, ir, extra=()):
    return ["apply", "--config", str(REPO / "configs/fusion" / config), "--selection", str(sel),
            "--rgb-preds", str(rgb), "--ir-preds", str(ir), "--data-root", str(s["root"]),
            "--meta", str(s["meta"]), "--split", "val", "--dataset", "dronevehicle",
            "--project", str(tmp_path / "e"), "--results", str(tmp_path / "res"),
            *extra]  # fmt: skip


def test_apply_variant_names_and_format(tmp_path: Path):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    sel = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    rgb = save_predictions(tmp_path / "E1_val_onnx_int8" / "predictions_coco.json.gz", RGB)
    ir = save_predictions(tmp_path / "E2_val_onnx_int8" / "predictions_coco.json.gz", IR)
    args = _dv_apply_args(tmp_path, s, sel, "e4_late_fusion.yaml", rgb, ir,
                          ["--variant", "onnx_int8", "--n-boot", "0"])  # fmt: skip
    assert main(args) == 0
    run_id = json.loads(sel.read_text(encoding="utf-8"))["run_id"]
    rec = json.loads((tmp_path / "res/edge" / f"{run_id}_val_onnx_int8_ir.json").read_text(
        encoding="utf-8"))  # fmt: skip
    assert rec["format"] == "onnx" and rec["precision"] == "int8"
    assert rec["fusion"]["variant"] == "onnx_int8" and rec["n_boot"] == 0
    assert rec["metrics"]["ci95"]["map50"] == [None, None]
    assert (tmp_path / "e" / f"{run_id}_val_onnx_int8" / "predictions_coco.json.gz").exists()


def test_apply_e4b_on_dronevehicle_uses_slices(tmp_path: Path):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    base = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    assert main(select_args(tmp_path, s, "e4b_late_fusion_illum.yaml",
                            ["--base", str(base)])) == 0  # fmt: skip
    sel = selection_path(tmp_path / "res", "e4b_late_fusion_illum", "val")
    rgb = save_predictions(tmp_path / "E1_val_onnx_fp32" / "predictions_coco.json.gz", RGB)
    ir = save_predictions(tmp_path / "E2_val_onnx_fp32" / "predictions_coco.json.gz", IR)
    args = _dv_apply_args(tmp_path, s, sel, "e4b_late_fusion_illum.yaml", rgb, ir,
                          ["--variant", "onnx_fp32", "--n-boot", "10"])  # fmt: skip
    assert main(args) == 0
    run_id = json.loads(sel.read_text(encoding="utf-8"))["run_id"]
    rec = json.loads((tmp_path / "res/edge" / f"{run_id}_val_onnx_fp32_ir.json").read_text(
        encoding="utf-8"))  # fmt: skip
    assert rec["fusion"]["ir_weight"] == {"day": 0.5, "night": 2.0}


def test_apply_e4b_without_slices_is_error(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    base = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    assert main(select_args(tmp_path, s, "e4b_late_fusion_illum.yaml",
                            ["--base", str(base)])) == 0  # fmt: skip
    (s["meta"] / "slices_val.json").unlink()
    sel = selection_path(tmp_path / "res", "e4b_late_fusion_illum", "val")
    rgb = save_predictions(tmp_path / "E1_val_x" / "predictions_coco.json.gz", RGB)
    ir = save_predictions(tmp_path / "E2_val_x" / "predictions_coco.json.gz", IR)
    args = _dv_apply_args(tmp_path, s, sel, "e4b_late_fusion_illum.yaml", rgb, ir,
                          ["--results-subdir", "t"])  # fmt: skip
    assert main(args) == 2 and "slices_val.json" in capsys.readouterr().err


# --- H5 dal incelemesi bulguları (2026-10-17) ---


def test_select_checks_sources_before_search(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    wrong = save_predictions(tmp_path / "E9_val" / "predictions_coco.json.gz", RGB)
    args = select_args(tmp_path, s, "e4_late_fusion.yaml")
    args[args.index(str(s["rgb"]))] = str(wrong)
    assert main(args) == 2 and "E1" in capsys.readouterr().err
    assert not selection_path(tmp_path / "res", "e4_late_fusion", "val").exists()


def test_select_resume_with_other_preds_is_error(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    other = save_predictions(tmp_path / "z" / "E1_val" / "predictions_coco.json.gz", RGB)
    args = select_args(tmp_path, s, "e4_late_fusion.yaml")
    args[args.index(str(s["rgb"]))] = str(other)
    assert main(args) == 2 and "seçilmiş" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("rgb_dir", "ir_dir", "extra"),
    [
        ("E1_test", "E2_test", []),  # yanlış bölüm (val isteniyor)
        ("E1_val_onnx_fp32", "E2_val_onnx_fp32", ["--variant", "onnx_int8"]),  # yanlış varyant
        ("E1_val_onnx_int8", "E2_val_onnx_fp32", ["--variant", "onnx_int8"]),  # karışık çift
    ],
)
def test_apply_requires_exact_prediction_folder(tmp_path: Path, capsys, rgb_dir, ir_dir, extra):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    sel = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    rgb = save_predictions(tmp_path / rgb_dir / "predictions_coco.json.gz", RGB)
    ir = save_predictions(tmp_path / ir_dir / "predictions_coco.json.gz", IR)
    args = _dv_apply_args(tmp_path, s, sel, "e4_late_fusion.yaml", rgb, ir, extra)
    assert main(args) == 2 and "klasör" in capsys.readouterr().err


def test_variant_record_stale_after_prediction_change_is_error(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    sel = selection_path(tmp_path / "res", "e4_late_fusion", "val")
    rgb = save_predictions(tmp_path / "E1_val_onnx_int8" / "predictions_coco.json.gz", RGB)
    ir = save_predictions(tmp_path / "E2_val_onnx_int8" / "predictions_coco.json.gz", IR)
    args = _dv_apply_args(tmp_path, s, sel, "e4_late_fusion.yaml", rgb, ir,
                          ["--variant", "onnx_int8", "--n-boot", "0"])  # fmt: skip
    assert main(args) == 0
    assert main(args) == 0  # aynı girdi: atlanır
    save_predictions(rgb, RGB[:1])  # model yeniden nicemlendi: aynı yola yeni tahminler
    assert main(args) == 2 and "farklı girdilerle" in capsys.readouterr().err


def test_old_records_without_prediction_hashes_still_skip(tmp_path: Path, capsys):
    s = setup_dv(tmp_path)
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    for p in (tmp_path / "res/runs").glob("*.json"):  # H5 öncesi kayıtlar: sha256 alanı yok
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["fusion"].pop("rgb_preds_sha256", None)
        rec["fusion"].pop("ir_preds_sha256", None)
        p.write_text(json.dumps(rec), encoding="utf-8")
    assert main(select_args(tmp_path, s, "e4_late_fusion.yaml")) == 0
    assert "zaten var" in capsys.readouterr().out
