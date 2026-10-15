"""baykus-robust: sağlamlık paketi (spec §12.3, FR-V4, kanca K1).

Koşul koşul ilerler: bozulmuş klasörü üret → koşulun uygulandığı her modeli değerlendir →
klasörü sil. Sonuç JSON'u olan (model, koşul) atlanır: oturum koparsa aynı komut kaldığı
yerden sürer. Bütün modeller aynı seed'le bozulmuş aynı pikselleri görür.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from baykus.robustness.corruptions import SUITE, Condition, applies, suite_conditions
from baykus.robustness.dataset import make_condition
from baykus.train.config import MODALITY_DIR, ConfigError, ExperimentConfig, load_config

ROBUST_DIR = "robust"
LABEL_SET = "ir"  # ana etiket seti (ADR 0003); sağlamlık sadece onunla ölçülür


class RobustError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    config: str
    weights: Path
    cfg: ExperimentConfig
    run_id: str


def load_model_spec(arg: str) -> ModelSpec:
    from baykus.eval.cli import train_record

    if "=" not in arg:
        raise RobustError(f"--model CONFIG=AĞIRLIK biçiminde olmalı (= yok): {arg!r}")
    config, w = arg.split("=", 1)
    weights = Path(w)
    run_id = train_record(weights).get("run_id")
    if not run_id:
        raise RobustError(f"{weights}: eğitim kaydı (baykus_train.json) yok; run_id bilinmiyor")
    return ModelSpec(config, weights, load_config(Path(config)), run_id)


def result_path(results: Path, m: ModelSpec, split: str, cond: Condition) -> Path:
    return Path(results) / ROBUST_DIR / f"{m.run_id}_{split}_{cond.id}_{LABEL_SET}.json"


def _mismatch(p: Path, seed: int | None, n_boot: int | None) -> str | None:
    rec = json.loads(p.read_text(encoding="utf-8"))
    got = (rec.get("robustness") or {}).get("seed")
    if seed is not None and got != seed:
        return f"seed {got} ≠ {seed}"
    if n_boot is not None and rec.get("n_boot") != n_boot:
        return f"n_boot {rec.get('n_boot')} ≠ {n_boot}"
    return None


def plan(
    models: list[ModelSpec],
    conds: list[Condition],
    results: Path,
    split: str,
    *,
    seed: int | None = None,
    n_boot: int | None = None,
) -> list[tuple[Condition, list[ModelSpec]]]:
    """Koşulun uygulandığı ve sonucu olmayan modeller. Var olan sonuç başka seed/n_boot ile
    yazılmışsa hata: iki ayar aynı tabloda sessizce karışmasın."""
    out = []
    for c in conds:
        todo = []
        for m in models:
            if not applies(c, m.cfg.mode):
                continue
            p = result_path(results, m, split, c)
            if p.exists():
                why = _mismatch(p, seed, n_boot)
                if why:
                    raise RobustError(
                        f"{p.name}: önceki sonuç başka ayarla ({why}); aynı ayarı kullan "
                        "ya da dosyayı sil"
                    )
                continue
            todo.append(m)
        if todo:
            out.append((c, todo))
    return out


def run_suite(
    models: list[ModelSpec],
    conds: list[Condition],
    *,
    src: Path,
    work: Path,
    meta: Path,
    split: str,
    project: Path,
    results: Path,
    seed: int,
    n_boot: int,
    device: str | None = None,
    workers: int = 1,
    evaluate: Callable | None = None,
) -> int:
    if evaluate is None:
        from baykus.eval.cli import evaluate_weights

        evaluate = evaluate_weights
    n = 0
    for cond, todo in plan(models, conds, results, split, seed=seed, n_boot=n_boot):
        cdir = Path(work) / cond.id
        shutil.rmtree(cdir, ignore_errors=True)  # önceki oturumdan yarım kalmış olabilir
        mods = {MODALITY_DIR[m.cfg.mode] for m in todo}
        info = {
            "id": cond.id,
            "corruption": cond.corruption,
            "severity": cond.severity,
            "target": cond.target,
            "seed": seed,
        }
        try:
            make_condition(src, cdir, split, cond, mods, seed, workers, label_set=LABEL_SET)
            for m in todo:
                evaluate(
                    m.cfg,
                    config=m.config,
                    weights=m.weights,
                    data=cdir / Path(m.cfg.data).name,
                    split=split,
                    meta=meta,
                    project=project,
                    results=results,
                    n_boot=n_boot,
                    device=device,
                    label_sets=(LABEL_SET,),
                    condition=info,
                    results_subdir=ROBUST_DIR,
                )
                # Ultralytics'in ham tahmin kopyası Drive'da yer kaplar; bizimki .json.gz'de (K1)
                (Path(project) / f"{m.run_id}_{split}_{cond.id}" / "predictions.json").unlink(
                    missing_ok=True
                )
                n += 1
                print(f"[{cond.id}] {m.cfg.experiment} tamam", flush=True)
        finally:
            shutil.rmtree(cdir, ignore_errors=True)
    return n


def _cmd_run(a: argparse.Namespace) -> int:
    try:
        models = [load_model_spec(s) for s in a.model]
        n = run_suite(
            models,
            suite_conditions(a.only),
            src=Path(a.data_root),
            work=Path(a.work),
            meta=Path(a.meta),
            split=a.split,
            project=Path(a.project),
            results=Path(a.results),
            seed=a.seed,
            n_boot=a.n_boot if a.n_boot is not None else models[0].cfg.eval.n_boot,
            device=a.device,
            workers=a.workers,
        )
    except (RobustError, ConfigError, FileNotFoundError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
    print(f"değerlendirilen (model, koşul): {n}")
    return 0


def _cmd_report(a: argparse.Namespace) -> int:
    from baykus.report.robust import load_robust, robust_markdown, robust_plot

    recs = load_robust(Path(a.robust), split=a.split)
    if not recs:
        print(f"hata: {a.robust} altında {a.split} sağlamlık sonucu yok", file=sys.stderr)
        return 2
    out, fig = Path(a.out), Path(a.fig)
    rel = Path(os.path.relpath(fig, out.parent)).as_posix()
    try:
        text = robust_markdown(recs, rel)
    except ValueError as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
    robust_plot(recs, fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-robust")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="sağlamlık paketini çalıştır (kaldığı yerden sürer)")
    r.add_argument("--model", action="append", required=True, help="CONFIG=AĞIRLIK, tekrarlanır")
    r.add_argument("--data-root", required=True, help="işlenmiş veri kökü (rgb/, ir/, coco/)")
    r.add_argument("--meta", required=True, help="slices_<split>.json klasörü")
    r.add_argument("--split", default="val", choices=["val", "test"])
    r.add_argument("--work", default="runs/robust_work", help="geçici koşul klasörleri")
    r.add_argument("--project", default="runs/robust", help="tahmin dosyaları (K1)")
    r.add_argument("--results", default="../results")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--only", nargs="+", choices=list(SUITE), default=None)
    r.add_argument("--n-boot", type=int, default=None)
    r.add_argument("--device", default=None)
    r.add_argument("--workers", type=int, default=1)
    r.set_defaults(func=_cmd_run)
    rp = sub.add_parser("report", help="results/robust → docs/robustness.md + eğriler")
    rp.add_argument("--robust", default="../results/robust")
    rp.add_argument("--out", default="../docs/robustness.md")
    rp.add_argument("--fig", default="../docs/figures/results/robustness.png")
    rp.add_argument("--split", default="val", choices=["val", "test"])
    rp.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)
