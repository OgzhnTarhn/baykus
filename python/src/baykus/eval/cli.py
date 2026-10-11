"""baykus-eval: Ultralytics val → pycocotools metrikleri + güven aralığı → run JSON (FR-V)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
from pathlib import Path

from pycocotools.coco import COCO

from baykus.eval.metrics import build_metrics
from baykus.eval.predictions import load_predictions, remap_to_gt, save_predictions
from baykus.runinfo import (
    git_info,
    iso,
    make_run_id,
    model_family,
    model_size,
    software_versions,
    utc_now,
    write_json,
)


def _load_coco(path: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def _hardware() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"device": torch.cuda.get_device_name(0), "driver": torch.version.cuda}
    except ImportError:
        pass
    return {"device": platform.processor() or platform.machine() or "cpu", "driver": None}


def _slices(meta: Path, split: str) -> dict | None:
    p = meta / f"slices_{split}.json"
    return json.loads(p.read_text(encoding="utf-8")).get("illumination") if p.exists() else None


def _cmd_run(a: argparse.Namespace) -> int:
    from ultralytics import YOLO

    from baykus.data.convert import activate_label_set
    from baykus.eval.coco_eval import MAX_DETS
    from baykus.train.cli import RECORD, resolve_data
    from baykus.train.config import load_config

    cfg = load_config(Path(a.config))
    weights = Path(a.weights)
    data = resolve_data(cfg, a.data_root)
    root = data.parent
    meta = Path(a.meta) if a.meta else root / "meta"
    rec_path = weights.parent.parent / RECORD
    train_rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
    git = git_info()
    run_id = train_rec.get("run_id") or make_run_id(
        cfg.experiment, cfg.model, git["git_commit"], utc_now()
    )

    activate_label_set(root, cfg.modality, "ir")  # çapraz kontrol IR etiketleriyle (FR-V6)
    name = f"{run_id}_{a.split}"
    kwargs = dict(
        data=str(data),
        split=a.split,
        conf=0.001,
        max_det=MAX_DETS,
        save_json=True,
        imgsz=cfg.train.imgsz,
        batch=cfg.train.batch,
        plots=False,
        project=str(a.project),
        name=name,
        exist_ok=True,
    )
    if a.device:
        kwargs["device"] = a.device
    ul = YOLO(str(weights)).val(**kwargs)
    save_dir = Path(a.project) / name
    pred_file = save_dir / "predictions.json"  # Ultralytics tahmin yoksa dosyayı hiç yazmaz
    ul_preds = json.loads(pred_file.read_text(encoding="utf-8")) if pred_file.exists() else []
    n_boot = a.n_boot if a.n_boot is not None else cfg.eval.n_boot

    for i, ls in enumerate(cfg.eval.label_sets):
        gt = _load_coco(root / "coco" / f"{a.split}_{ls}.json")
        dets, dropped = remap_to_gt(ul_preds, gt)
        if i == 0:
            save_predictions(save_dir / "predictions_coco.json.gz", dets)
        metrics = build_metrics(
            gt, dets, _slices(meta, a.split), n_boot=n_boot, seed=cfg.train.seed
        )
        record = {
            "run_id": run_id,
            "git_commit": git["git_commit"],
            "git_dirty": git["git_dirty"],
            "config": a.config,
            "config_hash": cfg.config_hash,
            "mlflow_run_id": train_rec.get("mlflow_run_id"),
            "train_run_id": train_rec.get("run_id"),
            "seed": cfg.train.seed,
            "dataset": "dronevehicle",
            "split": a.split,
            "label_set": ls,
            "model": {
                "family": model_family(cfg.model),
                "size": model_size(cfg.model),
                "mode": cfg.mode,
                "params_m": None,
                "gflops": None,
            },
            "format": "pytorch",
            "precision": "fp32",
            "hardware": _hardware(),
            "software": software_versions(),
            "weights": str(weights),
            "n_boot": n_boot,
            "dropped_predictions": dropped,
            "metrics": metrics,
            "created_at": iso(utc_now()),
        }
        if ls == "ir":
            ul_map50 = float(ul.box.map50)
            record["crosscheck"] = {
                "label_set": "ir",
                "ultralytics_map50": ul_map50,
                "pycocotools_map50": metrics["map50"],
                "abs_diff": abs(ul_map50 - metrics["map50"]),
            }
        out = write_json(Path(a.results) / "runs" / f"{name}_{ls}.json", record)
        print(out, f"map50={metrics['map50']:.4f}", f"dropped={dropped}")
        if dropped:
            print(f"uyarı: {dropped} tahmin eşlenemedi", file=sys.stderr)
    return 0


def _cmd_compare(a: argparse.Namespace) -> int:
    from baykus.eval.bootstrap import collect_matches, paired_diff_ci, subset

    gt = _load_coco(Path(a.gt))
    ma = collect_matches(gt, load_predictions(Path(a.a)))
    mb = collect_matches(gt, load_predictions(Path(a.b)))
    out = {
        "name": a.name,
        "gt": a.gt,
        "a": a.a,
        "b": a.b,
        "n_boot": a.n_boot,
        "overall": paired_diff_ci(ma, mb, a.n_boot),
        "per_illumination": {},
    }
    slices = json.loads(Path(a.slices).read_text(encoding="utf-8")).get("illumination", {})
    for name, ids in slices.items():
        out["per_illumination"][name] = (
            paired_diff_ci(subset(ma, ids), subset(mb, ids), a.n_boot) if ids else None
        )
    out["created_at"] = iso(utc_now())
    print(write_json(Path(a.results) / "compare" / f"{a.name}.json", out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="bir modeli değerlendir")
    r.add_argument("--config", required=True)
    r.add_argument("--weights", required=True)
    r.add_argument("--split", default="val", choices=["train", "val", "test"])
    r.add_argument("--data-root", type=Path, default=None)
    r.add_argument("--meta", type=Path, default=None)
    r.add_argument("--project", type=Path, default=Path("runs/eval"))
    r.add_argument("--results", type=Path, default=Path("../results"))
    r.add_argument("--n-boot", type=int, default=None)
    r.add_argument("--device", default=None)
    r.set_defaults(func=_cmd_run)
    c = sub.add_parser("compare", help="iki modelin eşleştirilmiş bootstrap farkı")
    c.add_argument("--gt", required=True)
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--slices", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--results", type=Path, default=Path("../results"))
    c.add_argument("--n-boot", type=int, default=1000)
    c.set_defaults(func=_cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)
