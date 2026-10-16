"""baykus-fusion: geç füzyon (FR-V7; E4, E4b; kanca K4). GPU gerekmez.

select — val'de WBF parametrelerini seçer; seçim JSON'u + birleşik tahminlerin run JSON'ları.
robust — sağlamlık paketi: E1 ve E2'nin koşul tahminleri birleştirilir (Görev 11).
apply  — seçilmiş parametrelerle başka bir veri setinde (M3OT, E7) birleştirir (Görev 11).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from baykus.eval.cli import (
    DEFAULT_DATASET,
    eval_record,
    load_coco,
    load_slices,
    model_block,
    run_name,
)
from baykus.eval.metrics import build_metrics
from baykus.eval.predictions import load_predictions, save_predictions
from baykus.fusion.config import FusionConfig, load_fusion_config
from baykus.fusion.late import fuse_predictions, illumination_weights, levels_from_slices
from baykus.fusion.select import select_constant, select_per_illumination
from baykus.runinfo import git_info, iso, make_run_id, utc_now, write_json
from baykus.train.config import ConfigError

SEL_DIR = "fusion"
LABEL_SET = "ir"  # seçim ölçütü ana etiket setinde (ADR 0003)


class FusionError(ValueError):
    pass


def selection_path(results: Path, experiment: str, split: str) -> Path:
    return Path(results) / SEL_DIR / f"{experiment}_{split}_selection.json"


def _weights(sel: dict, levels: dict[int, str] | None):
    s = sel["selected"]
    if isinstance(s["ir_weight"], dict):
        if levels is None:
            raise FusionError(
                "aydınlatmaya göre ağırlık için görüntülerin aydınlatma etiketi gerekli"
            )
        return illumination_weights(levels, s["ir_weight"], s["w_rgb"])
    return (s["w_rgb"], s["ir_weight"])


def fuse_and_record(
    sel: dict,
    cfg: FusionConfig,
    *,
    rgb_preds: Path,
    ir_preds: Path,
    root: Path,
    meta: Path,
    split: str,
    project: Path,
    results: Path,
    n_boot: int,
    workers: int = 1,
    dataset: str = DEFAULT_DATASET,
    class_agnostic: bool = False,
    condition: dict | None = None,
    levels: dict[int, str] | None = None,
    label_sets: tuple[str, ...] | None = None,
    results_subdir: str = "runs",
) -> list[Path]:
    label_sets = label_sets or cfg.label_sets
    name = run_name(sel["run_id"], split, dataset, condition)
    outs = [Path(results) / results_subdir / f"{name}_{ls}.json" for ls in label_sets]
    if all(p.exists() for p in outs):
        print(f"zaten var, atlandı: {name}")
        return outs
    root = Path(root)
    gt_ir = load_coco(root / "coco" / f"{split}_ir.json")
    images = {im["id"]: (im["width"], im["height"]) for im in gt_ir.dataset["images"]}
    weights = _weights(sel, levels)
    rgb, ir = load_predictions(Path(rgb_preds)), load_predictions(Path(ir_preds))
    if class_agnostic:
        rgb = [{**d, "category_id": 1} for d in rgb]
        ir = [{**d, "category_id": 1} for d in ir]
    s = sel["selected"]
    fused = fuse_predictions(rgb, ir, images, weights, iou_thr=s["iou_thr"],
                             skip_box_thr=sel["skip_box_thr"], workers=workers)  # fmt: skip
    save_predictions(Path(project) / name / "predictions_coco.json.gz", fused)
    info = {
        "rgb_run_id": sel["rgb_run_id"],
        "ir_run_id": sel["ir_run_id"],
        "rgb_preds": str(rgb_preds),
        "ir_preds": str(ir_preds),
        "iou_thr": s["iou_thr"],
        "w_rgb": s["w_rgb"],
        "ir_weight": s["ir_weight"],
        "skip_box_thr": sel["skip_box_thr"],
        "conf_type": "avg",
        "selected_on": sel["split"],
        "selection": sel["experiment"],
    }
    git = git_info()
    for ls, out in zip(label_sets, outs, strict=True):
        gt = load_coco(root / "coco" / f"{split}_{ls}.json")
        metrics = build_metrics(gt, fused, load_slices(meta, split), n_boot=n_boot, seed=cfg.seed)
        rec = eval_record(
            run_id=sel["run_id"],
            config=sel["config"],
            config_hash=sel["config_hash"],
            git=git,
            seed=cfg.seed,
            dataset=dataset,
            split=split,
            label_set=ls,
            model=model_block(cfg.model, "late_fusion"),
            weights=None,
            n_boot=n_boot,
            dropped=0,
            metrics=metrics,
            class_agnostic=class_agnostic,
        )
        rec["fusion"] = info
        if condition is not None:
            rec["robustness"] = condition
        write_json(out, rec)
        print(out, f"map50={metrics['map50']:.4f}", flush=True)
    return outs


def _cmd_select(a: argparse.Namespace) -> int:
    cfg = load_fusion_config(Path(a.config))
    root, meta = Path(a.data_root), Path(a.meta)
    out = selection_path(a.results, cfg.experiment, "val")
    if out.exists():
        sel = json.loads(out.read_text(encoding="utf-8"))
        print(f"seçim zaten var, yeniden aranmadı: {out}")
    else:
        gt = load_coco(root / "coco" / f"val_{LABEL_SET}.json")
        rgb, ir = load_predictions(Path(a.rgb_preds)), load_predictions(Path(a.ir_preds))
        t0 = time.monotonic()
        if cfg.per_illumination:
            if not a.base:
                raise FusionError("per_illumination: --base (E4 seçim JSON'u) gerekli")
            base = json.loads(Path(a.base).read_text(encoding="utf-8"))
            if (base["rgb_run_id"], base["ir_run_id"]) != (a.rgb_run_id, a.ir_run_id):
                raise FusionError("--base başka modellerle seçilmiş; aynı E1/E2 olmalı")
            slices = load_slices(meta, "val")
            if not slices:
                raise FusionError(f"{meta}/slices_val.json yok")
            found = select_per_illumination(rgb, ir, gt, cfg, slices,
                                            base["selected"]["iou_thr"], a.workers)  # fmt: skip
        else:
            found = select_constant(rgb, ir, gt, cfg, a.workers)
        git = git_info()
        sel = {
            "experiment": cfg.experiment,
            "config": a.config,
            "config_hash": cfg.config_hash,
            "run_id": make_run_id(cfg.experiment, cfg.model, git["git_commit"], utc_now()),
            "split": "val",
            "label_set": LABEL_SET,
            "criterion": "map50",
            "rgb_run_id": a.rgb_run_id,
            "ir_run_id": a.ir_run_id,
            "rgb_preds": str(a.rgb_preds),
            "ir_preds": str(a.ir_preds),
            "skip_box_thr": cfg.skip_box_thr,
            **found,
            "seconds": round(time.monotonic() - t0, 1),
            "created_at": iso(utc_now()),
        }
        write_json(out, sel)
        print(out, json.dumps(sel["selected"], ensure_ascii=False), f"{sel['seconds']} sn")
    levels = levels_from_slices(load_slices(meta, "val") or {}) if cfg.per_illumination else None
    fuse_and_record(sel, cfg, rgb_preds=Path(a.rgb_preds), ir_preds=Path(a.ir_preds), root=root,
                    meta=meta, split="val", project=Path(a.project), results=Path(a.results),
                    n_boot=a.n_boot or cfg.n_boot, workers=a.workers,
                    levels=levels)  # fmt: skip
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-fusion")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="val'de WBF parametrelerini seç (E4 / E4b)")
    s.add_argument("--config", required=True)
    s.add_argument("--rgb-preds", required=True)
    s.add_argument("--ir-preds", required=True)
    s.add_argument("--rgb-run-id", required=True)
    s.add_argument("--ir-run-id", required=True)
    s.add_argument("--data-root", required=True)
    s.add_argument("--meta", required=True)
    s.add_argument("--base", default=None, help="E4b için E4'ün seçim JSON'u")
    s.add_argument("--project", default="runs/eval")
    s.add_argument("--results", default="../results")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--n-boot", type=int, default=None)
    s.set_defaults(func=_cmd_select)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (FusionError, ConfigError, FileNotFoundError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
