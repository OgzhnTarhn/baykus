"""baykus-bench: bir pipeline × varyantın gecikmesi (spec §14) → results/bench/<P>_<V>_<H>.json.

Modeller `<models-dir>/<run_id>_<varyant>.onnx|.engine`; girdiler `baykus-edge bench-inputs`
klasörü. Aynı model dosyalarıyla (sha256) yazılmış sonuç varsa atlanır.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from baykus.bench.core import LateFusionPipeline, SinglePipeline, bench_record, time_pipeline
from baykus.edge.backends import open_backend
from baykus.edge.config import load_edge_config, repo_root
from baykus.edge.export import sha256_file
from baykus.edge.frames import load_input
from baykus.edge.variants import VARIANTS, model_file
from baykus.runinfo import write_json
from baykus.train.config import load_config

HARDWARE = ("laptop-cpu", "colab-t4")


def _cmd_run(a: argparse.Namespace) -> int:
    config = Path(a.config)
    cfg = load_edge_config(config)
    pipe = cfg.pipelines[a.pipeline]
    root = repo_root(config)
    names = [pipe.model] if pipe.kind == "single" else [pipe.rgb, pipe.ir]
    files = {n: model_file(Path(a.models_dir), cfg.models[n].run_id, a.variant) for n in names}
    models = [{"name": n, "file": str(f), "size_mb": round(f.stat().st_size / 1e6, 3),
               "sha256": sha256_file(f)} for n, f in files.items()]  # fmt: skip
    out = Path(a.results) / "bench" / f"{a.pipeline}_{a.variant}_{a.hardware}.json"
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if [m["sha256"] for m in old["models"]] == [m["sha256"] for m in models]:
            print(f"zaten var, atlandı: {out}")
            return 0
        raise ValueError(f"başka model dosyalarıyla ölçülmüş sonuç var: {out}; silip yeniden ölç")
    inputs = Path(a.inputs)
    index = json.loads((inputs / "images.json").read_text(encoding="utf-8"))

    def frames(name: str) -> list:
        mod = load_config(root / cfg.models[name].config).modality
        return [load_input(inputs, im["split"], im["stem"], mod) for im in index]

    backends = {n: open_backend(f, a.threads) for n, f in files.items()}
    if pipe.kind == "single":
        p = SinglePipeline(backends[pipe.model], frames(pipe.model), cfg.bench.conf)
    else:
        sel = json.loads((root / pipe.selection).read_text(encoding="utf-8"))
        pairs = list(zip(frames(pipe.rgb), frames(pipe.ir), strict=True))
        p = LateFusionPipeline(backends[pipe.rgb], backends[pipe.ir], pairs, sel, cfg.bench.conf)
    warmup = a.warmup if a.warmup is not None else cfg.bench.warmup
    runs = a.runs if a.runs is not None else cfg.bench.runs
    times = time_pipeline(p.step, warmup, runs)
    first = next(iter(backends.values()))
    protocol = {"warmup": warmup, "runs": runs, "imgsz": first.imgsz, "images": len(index),
                "conf": cfg.bench.conf, "provider": first.provider}  # fmt: skip
    rec = bench_record(pipeline=a.pipeline, variant=a.variant, hardware=a.hardware, models=models,
                       times=times, protocol=protocol, threads=a.threads)  # fmt: skip
    write_json(out, rec)
    t = rec["stages"]["total"]
    print(f"{a.pipeline} {a.variant} {a.hardware}: p50 {t['p50']:.1f} ms · p95 {t['p95']:.1f} ms"
          f" · {rec['fps']:.1f} FPS -> {out}")  # fmt: skip
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="bir pipeline × varyantın gecikmesini ölç")
    r.add_argument("--config", required=True)
    r.add_argument("--pipeline", required=True)
    r.add_argument("--variant", required=True, choices=list(VARIANTS))
    r.add_argument("--models-dir", required=True)
    r.add_argument("--inputs", required=True, help="baykus-edge bench-inputs --out klasörü")
    r.add_argument("--hardware", required=True, choices=HARDWARE)
    r.add_argument("--threads", type=int, default=None, help="verilmezse ORT varsayılanı")
    r.add_argument("--warmup", type=int, default=None, help="config'teki değerin yerine")
    r.add_argument("--runs", type=int, default=None, help="config'teki değerin yerine")
    r.add_argument("--results", default="../results")
    r.set_defaults(func=_cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
