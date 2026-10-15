"""baykus-eval: Ultralytics val → pycocotools metrikleri + güven aralığı → run JSON (FR-V)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from pycocotools.coco import COCO

from baykus.eval.metrics import build_metrics
from baykus.eval.predictions import load_predictions, remap_to_gt, save_predictions
from baykus.runinfo import (
    git_info,
    hardware_info,
    iso,
    make_run_id,
    model_family,
    model_size,
    software_versions,
    utc_now,
    write_json,
)
from baykus.train.config import ExperimentConfig

DEFAULT_DATASET = "dronevehicle"


def load_coco(path: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def load_slices(meta: Path, split: str) -> dict | None:
    p = Path(meta) / f"slices_{split}.json"
    return json.loads(p.read_text(encoding="utf-8")).get("illumination") if p.exists() else None


def run_name(
    run_id: str, split: str, dataset: str = DEFAULT_DATASET, condition: dict | None = None
) -> str:
    """Tahmin klasörü ve sonuç dosyası adı. DroneVehicle'da veri seti eki yok (H3 adları)."""
    ds = "" if dataset == DEFAULT_DATASET else f"_{dataset}"
    return f"{run_id}{ds}_{split}" + (f"_{condition['id']}" if condition else "")


def model_block(model: str, mode: str) -> dict:
    return {
        "family": model_family(model),
        "size": model_size(model),
        "mode": mode,
        "params_m": None,
        "gflops": None,
    }


def eval_record(
    *,
    run_id: str,
    config: str,
    config_hash: str,
    git: dict,
    seed: int,
    dataset: str,
    split: str,
    label_set: str,
    model: dict,
    weights: str | None,
    n_boot: int,
    dropped: int,
    metrics: dict,
    train_rec: dict | None = None,
    class_agnostic: bool = False,
) -> dict:
    """Spec §12.4 run JSON gövdesi; baykus-eval ve baykus-fusion ortak kullanır."""
    train_rec = train_rec or {}
    return {
        "run_id": run_id,
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "config": config,
        "config_hash": config_hash,
        "mlflow_run_id": train_rec.get("mlflow_run_id"),
        "train_run_id": train_rec.get("run_id"),
        "seed": seed,
        "dataset": dataset,
        "class_agnostic": class_agnostic,
        "split": split,
        "label_set": label_set,
        "model": model,
        "format": "pytorch",
        "precision": "fp32",
        "hardware": hardware_info(),
        "software": software_versions(),
        "weights": weights,
        "n_boot": n_boot,
        "dropped_predictions": dropped,
        "metrics": metrics,
        "created_at": iso(utc_now()),
    }


def train_record(weights: Path) -> dict:
    """<run>/weights/best.pt → <run>/baykus_train.json (yoksa {})."""
    from baykus.train.cli import RECORD

    p = Path(weights).parent.parent / RECORD
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def evaluate_weights(
    cfg: ExperimentConfig,
    *,
    config: str,
    weights: Path,
    data: Path,
    split: str,
    meta: Path,
    project: Path,
    results: Path,
    n_boot: int,
    device: str | None = None,
    label_sets: tuple[str, ...] | None = None,
    condition: dict | None = None,
    results_subdir: str = "runs",
    dataset: str = DEFAULT_DATASET,
    class_agnostic: bool = False,
) -> list[Path]:
    """Ultralytics val → etiket seti başına run JSON. condition verilirse (sağlamlık paketi)
    ada koşul kimliği eklenir ve kayda `robustness` alanı yazılır."""
    from ultralytics import YOLO

    from baykus.data.convert import activate_label_set
    from baykus.eval.coco_eval import MAX_DETS

    weights, data, meta = Path(weights), Path(data), Path(meta)
    root = data.parent
    train_rec = train_record(weights)
    git = git_info()
    run_id = train_rec.get("run_id") or make_run_id(
        cfg.experiment, cfg.model, git["git_commit"], utc_now()
    )

    activate_label_set(root, cfg.modality, "ir")  # çapraz kontrol IR etiketleriyle (FR-V6)
    name = run_name(run_id, split, dataset, condition)
    kwargs = dict(
        data=str(data),
        split=split,
        conf=0.001,
        max_det=MAX_DETS,
        save_json=True,
        imgsz=cfg.train.imgsz,
        batch=cfg.train.batch,
        plots=False,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    if class_agnostic:
        # single_cls: tahmin ve etiket sınıfı 0. agnostic_nms: YOLO26 uçtan uca başlığında top-k
        # sınıftan bağımsız seçilsin; yoksa aynı araç için ikinci sınıf kutusu FP sayılır.
        kwargs.update(single_cls=True, agnostic_nms=True)
    if device:
        kwargs["device"] = device
    save_dir = Path(project) / name
    # Önceki koşudan kalan dosya, bu koşu tahmin üretmezse yanlışlıkla okunurdu
    (save_dir / "predictions.json").unlink(missing_ok=True)
    ul = YOLO(str(weights)).val(**kwargs)
    pred_file = save_dir / "predictions.json"  # Ultralytics tahmin yoksa dosyayı hiç yazmaz
    ul_preds = json.loads(pred_file.read_text(encoding="utf-8")) if pred_file.exists() else []

    outs = []
    for i, ls in enumerate(label_sets or cfg.eval.label_sets):
        gt = load_coco(root / "coco" / f"{split}_{ls}.json")
        dets, dropped = remap_to_gt(ul_preds, gt)
        if class_agnostic:
            dets = [{**d, "category_id": 1} for d in dets]
        if i == 0:
            save_predictions(save_dir / "predictions_coco.json.gz", dets)
        metrics = build_metrics(
            gt, dets, load_slices(meta, split), n_boot=n_boot, seed=cfg.train.seed
        )
        record = eval_record(
            run_id=run_id,
            config=config,
            config_hash=cfg.config_hash,
            git=git,
            seed=cfg.train.seed,
            dataset=dataset,
            split=split,
            label_set=ls,
            model=model_block(cfg.model, cfg.mode),
            weights=str(weights),
            n_boot=n_boot,
            dropped=dropped,
            metrics=metrics,
            train_rec=train_rec,
            class_agnostic=class_agnostic,
        )
        if condition is not None:
            record["robustness"] = condition
        if ls == "ir":
            ul_map50 = float(ul.box.map50)
            record["crosscheck"] = {
                "label_set": "ir",
                "ultralytics_map50": ul_map50,
                "pycocotools_map50": metrics["map50"],
                "abs_diff": abs(ul_map50 - metrics["map50"]),
            }
        out = write_json(Path(results) / results_subdir / f"{name}_{ls}.json", record)
        print(out, f"map50={metrics['map50']:.4f}", f"dropped={dropped}")
        if dropped:
            print(f"uyarı: {dropped} tahmin eşlenemedi", file=sys.stderr)
        outs.append(out)
    return outs


def _cmd_run(a: argparse.Namespace) -> int:
    from baykus.train.cli import resolve_data
    from baykus.train.config import load_config

    cfg = load_config(Path(a.config))
    data = resolve_data(cfg, a.data_root)
    evaluate_weights(
        cfg,
        config=a.config,
        weights=Path(a.weights),
        data=data,
        split=a.split,
        meta=Path(a.meta) if a.meta else data.parent / "meta",
        project=a.project,
        results=a.results,
        n_boot=a.n_boot if a.n_boot is not None else cfg.eval.n_boot,
        device=a.device,
        dataset=a.dataset,
        class_agnostic=a.class_agnostic,
        results_subdir=a.results_subdir or ("runs" if a.dataset == DEFAULT_DATASET else a.dataset),
    )
    return 0


def _cmd_compare(a: argparse.Namespace) -> int:
    from baykus.eval.bootstrap import collect_matches, paired_mean_diff_ci, subset

    if len(a.a) != len(a.b):
        print("hata: --a ve --b aynı sayıda tahmin dosyası almalı (koşul koşul eşleşir)",
              file=sys.stderr)  # fmt: skip
        return 2
    gt = load_coco(Path(a.gt))
    ma = [collect_matches(gt, load_predictions(Path(p))) for p in a.a]
    mb = [collect_matches(gt, load_predictions(Path(p))) for p in a.b]

    def one(xs: list[str]) -> str | list[str]:
        return xs[0] if len(xs) == 1 else xs

    out = {
        "name": a.name,
        "split": a.split,
        "gt": a.gt,
        "a": one(a.a),
        "b": one(a.b),
        "n_boot": a.n_boot,
        "overall": paired_mean_diff_ci(ma, mb, a.n_boot),
        "per_illumination": {},
    }
    slices = json.loads(Path(a.slices).read_text(encoding="utf-8")).get("illumination", {})
    for name, ids in slices.items():
        sa, sb = [subset(m, ids) for m in ma], [subset(m, ids) for m in mb]
        # GT'de görüntüsü olmayan dilim ölçülmemiştir: 0 değil null
        out["per_illumination"][name] = (
            paired_mean_diff_ci(sa, sb, a.n_boot) if sa[0].img_ids else None
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
    r.add_argument("--dataset", default=DEFAULT_DATASET, choices=[DEFAULT_DATASET, "m3ot"])
    r.add_argument("--class-agnostic", action="store_true", help="sınıfsız AP (M3OT, spec §8.3)")
    r.add_argument("--results-subdir", default=None)
    r.set_defaults(func=_cmd_run)
    c = sub.add_parser("compare", help="iki modelin eşleştirilmiş bootstrap farkı")
    c.add_argument("--gt", required=True)
    c.add_argument("--a", required=True, nargs="+")
    c.add_argument("--b", required=True, nargs="+")
    c.add_argument("--slices", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--split", default="val", choices=["val", "test"])
    c.add_argument("--results", type=Path, default=Path("../results"))
    c.add_argument("--n-boot", type=int, default=1000)
    c.set_defaults(func=_cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)
