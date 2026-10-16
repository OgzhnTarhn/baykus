"""Varyant doğruluğu: ONNX/TensorRT modellerini COCO metrikleriyle değerlendirir (H5, FR-X4)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from baykus.edge.backends import open_backend
from baykus.edge.export import sha256_file
from baykus.edge.frames import load_input, postprocess, preprocess
from baykus.edge.variants import VARIANTS
from baykus.eval.cli import eval_record, load_coco, load_slices, model_block, run_name
from baykus.eval.metrics import build_metrics
from baykus.eval.predictions import save_predictions
from baykus.runinfo import git_info, write_json
from baykus.train.config import ExperimentConfig


class EdgeError(ValueError):
    """Varyant değerlendirmesinde tutarsız durum (farklı model ya da n_boot)."""


def predict_images(
    backend,
    root: Path,
    split: str,
    modality: str,
    images: list[dict],
    conf: float = 0.001,
) -> tuple[list[dict], float]:
    """Görüntüleri arka uçla çalıştırır; (COCO tahminleri, görüntü başına saniye) döner."""
    dets: list[dict] = []
    t0 = time.perf_counter()
    for image in images:
        img = load_input(root, split, Path(image["file_name"]).stem, modality)
        x, lb = preprocess(img, backend.imgsz)
        x = x.astype(backend.input_dtype, copy=False)
        out = backend.infer(x)
        dets.extend(postprocess(out, lb, conf, image["id"]))
    elapsed = time.perf_counter() - t0
    return dets, (elapsed / len(images) if images else 0.0)


def evaluate_variant(
    *,
    model_cfg: ExperimentConfig,
    run_id: str,
    config: str,
    variant: str,
    model_path: Path,
    data_root: Path,
    meta: Path,
    split: str,
    project: Path,
    results: Path,
    n_boot: int = 0,
    threads: int | None = None,
    label_sets: tuple[str, ...] = ("ir",),
    workers: int = 1,
    results_subdir: str = "edge",
) -> list[Path]:
    """Bir varyantı değerlendirir; etiket seti başına run JSON yazar. Aynı sha256 ve n_boot
    ile yazılmış sonuçlar atlanır, farklıysa EdgeError yükseltilir."""
    v = VARIANTS[variant]
    model_path, data_root, meta = Path(model_path), Path(data_root), Path(meta)
    name = run_name(run_id, split, variant=variant)
    outs = [Path(results) / results_subdir / f"{name}_{ls}.json" for ls in label_sets]
    sha = sha256_file(model_path)
    if all(p.exists() for p in outs):
        for p in outs:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec["n_boot"] != n_boot or rec["edge"]["sha256"] != sha:
                raise EdgeError(
                    f"farklı model ya da n_boot ile yazılmış sonuç var: {p}; "
                    "silip yeniden çalıştırın"
                )
        for p in outs:
            print(f"zaten var, atlandı: {p}")
        return outs

    gt_ir = load_coco(data_root / "coco" / f"{split}_ir.json")
    images = gt_ir.dataset["images"]
    backend = open_backend(model_path, threads=threads)
    dets, spi = predict_images(backend, data_root, split, model_cfg.modality, images)
    save_predictions(Path(project) / name / "predictions_coco.json.gz", dets)
    git = git_info()
    for ls, out in zip(label_sets, outs, strict=True):
        gt = load_coco(data_root / "coco" / f"{split}_{ls}.json")
        metrics = build_metrics(
            gt,
            dets,
            load_slices(meta, split),
            n_boot=n_boot,
            seed=model_cfg.train.seed,
            workers=workers,
        )
        rec = eval_record(
            run_id=run_id,
            config=config,
            config_hash=model_cfg.config_hash,
            git=git,
            seed=model_cfg.train.seed,
            dataset="dronevehicle",
            split=split,
            label_set=ls,
            model=model_block(model_cfg.model, model_cfg.mode),
            weights=str(model_path),
            n_boot=n_boot,
            dropped=0,
            metrics=metrics,
            fmt=v.fmt,
            precision=v.precision,
        )
        rec["edge"] = {
            "variant": variant,
            "file_size_mb": round(model_path.stat().st_size / 1e6, 3),
            "sha256": sha,
            "provider": backend.provider,
            "threads": backend.threads,
            "seconds_per_image": round(spi, 5),
        }
        write_json(out, rec)
        print(out, f"map50={metrics['map50']:.4f}", flush=True)
    return outs
