"""baykus-edge: kalibrasyon listesi ve ONNX/TensorRT dışa aktarımı (H5).

calib  — INT8 kalibrasyon görüntü adları + Ultralytics veri YAML'ı (isteğe bağlı rgbt4 üretimi).
export — seçilen varyantları dışa aktarır; manifest sayesinde yeniden başlatılabilir.
eval   — bir varyantı COCO metrikleriyle değerlendirir (FR-X4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baykus.data.multichannel import build_rgbt4
from baykus.edge.calib import calibration_stems, write_calib_data
from baykus.edge.config import load_edge_config, repo_root
from baykus.edge.evaluate import evaluate_variant
from baykus.edge.export import export_model
from baykus.edge.variants import VARIANTS
from baykus.train.config import load_config


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
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:  # ConfigError dahil
        print(f"hata: {e}", file=sys.stderr)
        return 2
