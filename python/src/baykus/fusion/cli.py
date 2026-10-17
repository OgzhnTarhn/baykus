"""baykus-fusion: geç füzyon (FR-V7; E4, E4b; kanca K4). GPU gerekmez.

select — val'de WBF parametrelerini seçer; seçim JSON'u + birleşik tahminlerin run JSON'ları.
robust — sağlamlık paketi: E1 ve E2'nin koşul tahminleri birleştirilir (Görev 11).
apply  — seçilmiş parametrelerle başka bir veri setinde (M3OT, E7) birleştirir (Görev 11).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from baykus.edge.variants import VARIANTS
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
from baykus.fusion.late import (
    condition_levels,
    fuse_predictions,
    illumination_weights,
    levels_from_slices,
    load_brightness,
)
from baykus.fusion.select import select_constant, select_per_illumination
from baykus.robustness.cli import ROBUST_DIR
from baykus.robustness.corruptions import SUITE, Condition, applies, suite_conditions
from baykus.runinfo import git_info, iso, make_run_id, utc_now, write_json

SEL_DIR = "fusion"
LABEL_SET = "ir"  # seçim ölçütü ana etiket setinde (ADR 0003)


class FusionError(ValueError):
    pass


def selection_path(results: Path, experiment: str, split: str) -> Path:
    return Path(results) / SEL_DIR / f"{experiment}_{split}_selection.json"


def check_sources(
    sel: dict,
    rgb_preds: Path,
    ir_preds: Path,
    *,
    split: str,
    dataset: str = DEFAULT_DATASET,
    variant: str | None = None,
    sources: dict | None = None,
) -> None:
    """Tahmin dosyası seçimdeki modelin, bu bölümün ve bu varyantın mı? Değerlendirme klasörü
    tam olarak baykus.eval.cli.run_name adını taşımalı (sağlamlıkta koşul eki `sources`'tan)."""
    for side, path in (("rgb", rgb_preds), ("ir", ir_preds)):
        run_id, folder = sel[f"{side}_run_id"], Path(path).parent.name
        cond = {"id": sources[side]} if sources else None
        want = run_name(run_id, split, dataset, cond, variant=variant)
        if folder != want:
            raise FusionError(
                f"{side} tahmini ({path}) beklenen klasörde değil: {want} "
                f"(model {run_id}, bölüm {split}, varyant {variant or 'pytorch'})"
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_inputs(path: Path, n_boot: int, rgb_preds: Path, ir_preds: Path) -> bool:
    rec = json.loads(path.read_text(encoding="utf-8"))
    fusion = rec.get("fusion") or {}
    same = (rec.get("n_boot"), fusion.get("rgb_preds"), fusion.get("ir_preds")) == (
        n_boot, str(rgb_preds), str(ir_preds)
    )  # fmt: skip
    # Aynı yola yeniden yazılmış tahmin (ör. yeniden nicemlenen model) özetten yakalanır;
    # H5 öncesi kayıtlarda özet yok, onlar yalnız yol ve n_boot ile karşılaştırılır.
    for side, p in (("rgb", rgb_preds), ("ir", ir_preds)):
        old = fusion.get(f"{side}_preds_sha256")
        if old is not None and old != _sha256(p):
            same = False
    return same


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
    variant: str | None = None,
) -> list[Path]:
    label_sets = label_sets or cfg.label_sets
    name = run_name(sel["run_id"], split, dataset, condition, variant=variant)
    outs = [Path(results) / results_subdir / f"{name}_{ls}.json" for ls in label_sets]
    check_sources(sel, rgb_preds, ir_preds, split=split, dataset=dataset, variant=variant,
                  sources=(condition or {}).get("sources"))  # fmt: skip
    if all(p.exists() for p in outs):
        stale = [p for p in outs if not _same_inputs(p, n_boot, rgb_preds, ir_preds)]
        if stale:
            raise FusionError(
                f"farklı girdilerle yazılmış sonuç var: {stale[0]}; silip yeniden çalıştırın"
            )
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
        "rgb_preds_sha256": _sha256(rgb_preds),
        "ir_preds_sha256": _sha256(ir_preds),
        "iou_thr": s["iou_thr"],
        "w_rgb": s["w_rgb"],
        "ir_weight": s["ir_weight"],
        "skip_box_thr": sel["skip_box_thr"],
        "conf_type": "avg",
        "selected_on": sel["split"],
        "selection": sel["experiment"],
        "variant": variant,
    }
    git = git_info()
    for ls, out in zip(label_sets, outs, strict=True):
        gt = load_coco(root / "coco" / f"{split}_{ls}.json")
        metrics = build_metrics(gt, fused, load_slices(meta, split), n_boot=n_boot, seed=cfg.seed,
                                workers=workers)  # fmt: skip
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
            fmt=VARIANTS[variant].fmt if variant else "pytorch",
            precision=VARIANTS[variant].precision if variant else "fp32",
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
    ids = {"rgb_run_id": a.rgb_run_id, "ir_run_id": a.ir_run_id}
    check_sources(ids, Path(a.rgb_preds), Path(a.ir_preds), split="val")  # aramadan önce
    if out.exists():
        sel = json.loads(out.read_text(encoding="utf-8"))
        if (sel["rgb_run_id"], sel["ir_run_id"], sel["config_hash"], sel["rgb_preds"],
                sel["ir_preds"]) != (a.rgb_run_id, a.ir_run_id, cfg.config_hash,
                                     str(a.rgb_preds), str(a.ir_preds)):  # fmt: skip
            raise FusionError(
                f"{out} başka modellerle, tahminlerle ya da config ile seçilmiş "
                f"({sel['rgb_run_id']}, {sel['ir_run_id']}); dosyayı taşıyıp yeniden seç"
            )
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
                    n_boot=_n_boot(a, cfg), workers=a.workers,
                    levels=levels)  # fmt: skip
    return 0


def compose_sources(cond: Condition) -> tuple[str, str]:
    """Geç füzyonun koşul girdileri: bozulma bir modaliteye dokunmuyorsa o modelin temiz
    tahmini. Hizasızlık IR'yi kaydırır: E2'nin kaydırılmış IR tahmini (--ir-misalign)."""
    rgb = cond.id if applies(cond, "rgb") else "clean"
    ir = cond.id if applies(cond, "ir") or cond.corruption == "misalign" else "clean"
    return rgb, ir


def _load_selection(a: argparse.Namespace) -> tuple[dict, FusionConfig]:
    cfg = load_fusion_config(Path(a.config))
    sel = json.loads(Path(a.selection).read_text(encoding="utf-8"))
    if sel["experiment"] != cfg.experiment:
        raise FusionError(f"seçim {sel['experiment']!r}, config {cfg.experiment!r}: eşleşmiyor")
    if sel["config_hash"] != cfg.config_hash:
        raise FusionError(
            f"seçim başka bir config ile yapılmış (config_hash {sel['config_hash'][:15]}… ≠ "
            f"{cfg.config_hash[:15]}…)"
        )
    return sel, cfg


def _cmd_robust(a: argparse.Namespace) -> int:
    sel, cfg = _load_selection(a)
    root, meta, rp = Path(a.data_root), Path(a.meta), Path(a.robust_preds)
    conds = [c for c in suite_conditions(a.only) if applies(c, "late_fusion")]
    srcs = {}
    for c in conds:
        cid_rgb, cid_ir = compose_sources(c)
        srcs[c.id] = (
            rp / f"{sel['rgb_run_id']}_{a.split}_{cid_rgb}" / "predictions_coco.json.gz",
            rp / f"{sel['ir_run_id']}_{a.split}_{cid_ir}" / "predictions_coco.json.gz",
            {"rgb": cid_rgb, "ir": cid_ir},
        )
    missing = sorted({str(p) for r, i, _ in srcs.values() for p in (r, i) if not p.exists()})
    if missing:
        raise FusionError(
            "tahmin dosyası yok (03_robust bitti mi? E2 hizasızlığı için --ir-misalign): "
            + ", ".join(missing)
        )
    brightness = th = None
    if cfg.per_illumination:
        gt = load_coco(root / "coco" / f"{a.split}_ir.json")
        brightness, th = load_brightness(meta, a.split, gt.dataset["images"])
    for c in conds:
        rgb_p, ir_p, sources = srcs[c.id]
        info = {"id": c.id, "corruption": c.corruption, "severity": c.severity,
                "target": c.target, "seed": a.seed, "sources": sources}  # fmt: skip
        fuse_and_record(
            sel, cfg, rgb_preds=rgb_p, ir_preds=ir_p, root=root, meta=meta, split=a.split,
            project=Path(a.project), results=Path(a.results), n_boot=_n_boot(a, cfg),
            workers=a.workers, condition=info, label_sets=(LABEL_SET,),
            results_subdir=ROBUST_DIR,
            levels=condition_levels(brightness, th, c) if cfg.per_illumination else None,
        )  # fmt: skip
    return 0


def _n_boot(a: argparse.Namespace, cfg: FusionConfig) -> int:
    return a.n_boot if a.n_boot is not None else cfg.n_boot  # 0 geçerli: bootstrap'sız


def _cmd_apply(a: argparse.Namespace) -> int:
    sel, cfg = _load_selection(a)
    root = Path(a.data_root)
    meta = Path(a.meta) if a.meta else root / "meta"
    levels = None
    if isinstance(sel["selected"]["ir_weight"], dict):
        if a.dataset != DEFAULT_DATASET:
            raise FusionError(
                f"aydınlatmaya göre ağırlık (E4b) {a.dataset}'te uygulanmaz: aydınlatma etiketi yok"
            )
        slices = load_slices(meta, a.split)
        if not slices:
            raise FusionError(f"{meta}/slices_{a.split}.json yok: E4b aydınlatma etiketi ister")
        levels = levels_from_slices(slices)
    subdir = a.results_subdir or (
        a.dataset if a.dataset != DEFAULT_DATASET else "edge" if a.variant else "runs"
    )
    fuse_and_record(
        sel, cfg, rgb_preds=Path(a.rgb_preds), ir_preds=Path(a.ir_preds), root=root, meta=meta,
        split=a.split, project=Path(a.project), results=Path(a.results), n_boot=_n_boot(a, cfg),
        workers=a.workers, dataset=a.dataset, class_agnostic=a.class_agnostic,
        results_subdir=subdir, levels=levels, variant=a.variant,
    )  # fmt: skip
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
    r = sub.add_parser("robust", help="sağlamlık paketi: koşul tahminlerini birleştir")
    r.add_argument("--config", required=True)
    r.add_argument("--selection", required=True)
    r.add_argument("--robust-preds", required=True, help="03_robust'ın --project klasörü")
    r.add_argument("--data-root", required=True)
    r.add_argument("--meta", required=True)
    r.add_argument("--split", default="val", choices=["val", "test"])
    r.add_argument("--project", default="runs/robust")
    r.add_argument("--results", default="../results")
    r.add_argument("--only", nargs="+", choices=list(SUITE), default=None)
    r.add_argument("--seed", type=int, default=42, help="03_robust'ın bozulma seed'i (kayıt için)")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--n-boot", type=int, default=None)
    r.set_defaults(func=_cmd_robust)
    ap = sub.add_parser("apply", help="seçilmiş parametrelerle başka veri setinde birleştir")
    ap.add_argument("--config", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--rgb-preds", required=True)
    ap.add_argument("--ir-preds", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--dataset", default="m3ot", choices=[DEFAULT_DATASET, "m3ot"])
    ap.add_argument("--class-agnostic", action="store_true")
    ap.add_argument("--project", default="runs/m3ot")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--variant", default=None, choices=list(VARIANTS), help="edge varyantı")
    ap.add_argument("--results-subdir", default=None)
    ap.set_defaults(func=_cmd_apply)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError) as e:  # FusionError, ConfigError dahil
        print(f"hata: {e}", file=sys.stderr)
        return 2
