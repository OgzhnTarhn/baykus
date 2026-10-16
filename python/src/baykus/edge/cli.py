"""baykus-edge: kalibrasyon listesi ve ONNX/TensorRT dışa aktarımı (H5).

calib  — INT8 kalibrasyon görüntü adları + Ultralytics veri YAML'ı (isteğe bağlı rgbt4 üretimi).
export — seçilen varyantları dışa aktarır; manifest sayesinde yeniden başlatılabilir.
eval   — bir varyantı COCO metrikleriyle değerlendirir (FR-X4).
bench-inputs — bench için val'den seçili görüntüler, .npy'ler ve Colab tahmin referansları.
parity — laptop çıktısı Colab referansıyla aynı mı (FR-E6 toleransları).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from baykus.data.multichannel import build_rgbt4
from baykus.edge.backends import open_backend
from baykus.edge.calib import calibration_stems, write_calib_data
from baykus.edge.config import load_edge_config, repo_root
from baykus.edge.evaluate import evaluate_variant, predict_images
from baykus.edge.export import export_model
from baykus.edge.frames import load_input, preprocess
from baykus.edge.parity import compare_dets
from baykus.edge.variants import ONNX_VARIANTS, VARIANTS, model_file
from baykus.eval.cli import load_coco, run_name
from baykus.eval.predictions import load_predictions
from baykus.runinfo import write_json
from baykus.train.config import load_config

EXT = {"rgb": ".jpg", "ir": ".png", "rgbt4": ".tif"}


def _modalities(config: Path, cfg) -> dict[str, str]:
    """models anahtarı → modalite (rgb / ir / rgbt4), model config'lerinden."""
    root = repo_root(config)
    return {k: load_config(root / ref.config).modality for k, ref in cfg.models.items()}


def _cmd_calib(a: argparse.Namespace) -> int:
    cfg = load_edge_config(Path(a.config))
    root = Path(a.data_root)
    stems = calibration_stems(root, cfg.calibration.split, cfg.calibration.n, cfg.seed)
    if a.modality == "rgbt4" and a.build_rgbt4:
        build_rgbt4(root, [cfg.calibration.split], a.workers, stems=stems)
    y = write_calib_data(root, a.modality, stems, Path(a.out))
    print(y, len(stems))
    return 0


def _cmd_export(a: argparse.Namespace) -> int:
    cfg = load_edge_config(Path(a.config))
    model_cfg = load_config(repo_root(Path(a.config)) / cfg.models[a.model].config)
    modality = model_cfg.modality
    stems: list[str] = []
    calib_yaml: Path | None = None
    if a.calib_dir:
        calib_dir = Path(a.calib_dir)
        txt = calib_dir / f"calib_{modality}.txt"
        stems = [
            Path(line).stem for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        y = calib_dir / f"data_{modality}_calib.yaml"
        calib_yaml = y if y.exists() else None
    out = export_model(
        Path(a.weights),
        Path(a.out),
        cfg.models[a.model].run_id,
        modality,
        a.variants,
        imgsz=cfg.imgsz,
        root=Path(a.data_root) if a.data_root else None,
        stems=stems,
        calib_yaml=calib_yaml,
    )
    print(len(out), "varyant")
    return 0


def _cmd_eval(a: argparse.Namespace) -> int:
    cfg = load_edge_config(Path(a.config))
    ref = cfg.models[a.model]
    model_cfg = load_config(repo_root(Path(a.config)) / ref.config)
    evaluate_variant(
        model_cfg=model_cfg,
        run_id=ref.run_id,
        config=ref.config,
        variant=a.variant,
        model_path=Path(a.file),
        data_root=Path(a.data_root),
        meta=Path(a.meta),
        split=a.split,
        project=Path(a.project),
        results=Path(a.results),
        n_boot=a.n_boot,
        threads=a.threads,
        label_sets=tuple(a.label_sets),
        workers=a.workers,
    )
    return 0


def _cmd_bench_inputs(a: argparse.Namespace) -> int:
    config, root, out = Path(a.config), Path(a.data_root), Path(a.out)
    cfg = load_edge_config(config)
    mods = a.modalities or sorted(set(_modalities(config, cfg).values()))
    images = sorted(load_coco(root / "coco" / f"{a.split}_ir.json").dataset["images"],
                    key=lambda im: im["file_name"])  # fmt: skip
    rng = np.random.default_rng(cfg.seed)
    pick = sorted(rng.choice(len(images), size=min(cfg.bench.images, len(images)), replace=False))
    chosen = [images[i] for i in pick]
    for im in chosen:
        stem = Path(im["file_name"]).stem
        for m in mods:
            dst = out / m / "images" / a.split / f"{stem}{EXT[m]}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / m / "images" / a.split / f"{stem}{EXT[m]}", dst)
            npy = out / "npy" / m / f"{stem}.npy"
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, preprocess(load_input(root, a.split, stem, m), cfg.imgsz)[0])
    index = [{"stem": Path(im["file_name"]).stem, "image_id": im["id"], "width": im["width"],
              "height": im["height"], "split": a.split} for im in chosen]  # fmt: skip
    write_json(out / "images.json", index)
    ids = {im["id"] for im in chosen}
    n_refs = 0
    for ref in cfg.models.values():
        for v in a.variants:
            src = Path(a.preds_dir) / run_name(ref.run_id, a.split, variant=v)
            src = src / "predictions_coco.json.gz"
            if not src.exists():
                continue
            keep = [d for d in load_predictions(src)
                    if d["image_id"] in ids and d["score"] >= cfg.bench.conf]  # fmt: skip
            write_json(out / "refs" / f"{ref.run_id}_{v}.json", keep)
            n_refs += 1
    print(out, f"{len(chosen)} görüntü, {len(mods)} modalite, {n_refs} referans")
    return 0


def _cmd_parity(a: argparse.Namespace) -> int:
    config, inputs = Path(a.config), Path(a.inputs)
    cfg = load_edge_config(config)
    mods = _modalities(config, cfg)
    index = json.loads((inputs / "images.json").read_text(encoding="utf-8"))
    images = [{"id": im["image_id"], "file_name": im["stem"]} for im in index]
    split = index[0]["split"] if index else "val"
    report: dict[str, dict] = {}
    for key, ref in cfg.models.items():
        ref_file = inputs / "refs" / f"{ref.run_id}_{a.variant}.json"
        if not ref_file.exists():
            continue
        backend = open_backend(model_file(Path(a.models_dir), ref.run_id, a.variant), a.threads)
        dets = predict_images(backend, inputs, split, mods[key], images, conf=cfg.bench.conf)[0]
        refs = json.loads(ref_file.read_text(encoding="utf-8"))
        report[key] = compare_dets(dets, refs, conf=cfg.bench.conf)
        print(key, a.variant, json.dumps(report[key], ensure_ascii=False))
    if not report:
        raise ValueError(f"{inputs / 'refs'}: {a.variant} için referans yok")
    out = Path(a.results) / "bench" / f"parity_{a.hardware}_{a.variant}.json"
    write_json(out, {"variant": a.variant, "hardware": a.hardware, "models": report})
    print(out)
    return 0 if all(r["ok"] for r in report.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-edge")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calib", help="INT8 kalibrasyon listesi ve veri YAML'ı üret")
    c.add_argument("--config", required=True)
    c.add_argument("--data-root", required=True)
    c.add_argument("--modality", required=True, choices=["rgb", "ir", "rgbt4"])
    c.add_argument("--out", required=True)
    c.add_argument("--build-rgbt4", action="store_true", help="rgbt4 için TIFF'leri de üret")
    c.add_argument("--workers", type=int, default=1)
    c.set_defaults(func=_cmd_calib)
    e = sub.add_parser("export", help="ONNX/TensorRT varyantlarını dışa aktar")
    e.add_argument("--config", required=True)
    e.add_argument("--model", required=True, help="config'teki models anahtarı")
    e.add_argument("--weights", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--variants", nargs="+", required=True, choices=list(VARIANTS))
    e.add_argument("--data-root", default=None)
    e.add_argument("--calib-dir", default=None, help="calib komutunun --out klasörü")
    e.set_defaults(func=_cmd_export)
    v = sub.add_parser("eval", help="bir varyantı COCO metrikleriyle değerlendir")
    v.add_argument("--config", required=True)
    v.add_argument("--model", required=True, help="config'teki models anahtarı")
    v.add_argument("--variant", required=True, choices=list(VARIANTS))
    v.add_argument("--file", required=True)
    v.add_argument("--data-root", required=True)
    v.add_argument("--meta", required=True)
    v.add_argument("--split", default="val", choices=["val", "test"])
    v.add_argument("--threads", type=int, default=None)
    v.add_argument("--n-boot", type=int, default=0)
    v.add_argument("--label-sets", nargs="+", default=["ir"], choices=["ir", "rgb"])
    v.add_argument("--workers", type=int, default=1)
    v.add_argument("--project", default="runs/edge")
    v.add_argument("--results", default="../results")
    v.set_defaults(func=_cmd_eval)
    bi = sub.add_parser("bench-inputs", help="bench girdileri ve Colab referansları")
    bi.add_argument("--config", required=True)
    bi.add_argument("--data-root", required=True)
    bi.add_argument("--split", default="val", choices=["val", "test"])
    bi.add_argument("--preds-dir", required=True, help="eval'in --project klasörü")
    bi.add_argument("--out", required=True)
    bi.add_argument("--variants", nargs="+", default=list(ONNX_VARIANTS), choices=list(VARIANTS))
    bi.add_argument("--modalities", nargs="+", default=None, choices=list(EXT))
    bi.set_defaults(func=_cmd_bench_inputs)
    pa = sub.add_parser("parity", help="çıktı eşliği: bu makine ↔ Colab referansı")
    pa.add_argument("--config", required=True)
    pa.add_argument("--inputs", required=True)
    pa.add_argument("--models-dir", required=True)
    pa.add_argument("--variant", required=True, choices=list(VARIANTS))
    pa.add_argument("--hardware", required=True)
    pa.add_argument("--threads", type=int, default=None)
    pa.add_argument("--results", default="../results")
    pa.set_defaults(func=_cmd_parity)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:  # ConfigError dahil
        print(f"hata: {e}", file=sys.stderr)
        return 2
