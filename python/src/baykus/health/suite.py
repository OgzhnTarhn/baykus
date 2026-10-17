"""Kapılı sağlamlık paketi (Faz 2 tasarımı §6.1-6.2, plan F2 Görev 5).

Faz 1'in yolu (bozulmuş klasör → Ultralytics val → run JSON), her karede kapıyla: bozulmadan
sonra sağlık ölçülür, zararlı kamera sıfırlanır. Kapı bir koşulun hiçbir pikselini
değiştirmediyse yeniden değerlendirme yok: kapılı tahminler = Faz 1 tahminleri (Spec yorumları 3).
Koşul başına özet JSON'u en son yazılır; varsa koşul atlanır (oturum koparsa kaldığı yerden).
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path

from baykus.eval.cli import run_name
from baykus.health.gate import FLAG_ORDER, GATES, check_frame, load_thresholds
from baykus.robustness.cli import LABEL_SET, ModelSpec
from baykus.robustness.corruptions import Condition
from baykus.robustness.dataset import make_condition
from baykus.runinfo import write_json

HEALTH_DIR = "health"
GATE_DIR = "gate"


class HealthError(ValueError):
    pass


def summarize_health(frames: list[dict]) -> dict:
    def flags(cam: str) -> dict[str, int]:
        return {f: sum(f in fr[cam]["flags"] for fr in frames) for f in FLAG_ORDER}

    return {
        "n_images": len(frames),
        "gate": {g: sum(fr["gate"] == g for fr in frames) for g in GATES},
        "changed": sum(bool(fr["changed"]) for fr in frames),
        "rgb": flags("rgb"),
        "ir": flags("ir"),
        "misaligned": sum(bool(fr["pair"]["misaligned"]) for fr in frames),
    }


def summary_path(results: Path, run_id: str, split: str, cond_id: str) -> Path:
    return Path(results) / HEALTH_DIR / f"{run_id}_{split}_{cond_id}_health.json"


def gate_name(run_id: str, split: str, cond_id: str) -> str:
    """Kapılı tahmin klasörü: Faz 1 klasörünün (<run>_<split>_<koşul>) yanında, _gate ekiyle."""
    return run_name(run_id, split, condition={"id": f"{cond_id}_gate"})


def run_gated_suite(
    model: ModelSpec,
    conds: list[Condition],
    thresholds: Path,
    *,
    src: Path,
    work: Path,
    meta: Path,
    split: str,
    project: Path,
    results: Path,
    seed: int = 42,
    workers: int = 1,
    device: str | None = None,
    evaluate: Callable | None = None,
) -> int:
    if model.cfg.mode != "early_fusion":
        raise HealthError(
            f"kapı 4 kanallı modelde anlamlı (mode early_fusion); verilen: {model.cfg.mode}"
        )
    if evaluate is None:
        from baykus.eval.cli import evaluate_weights

        evaluate = evaluate_weights
    th = load_thresholds(Path(thresholds))
    hook = partial(check_frame, th=th)
    n = 0
    for cond in conds:
        sp = summary_path(results, model.run_id, split, cond.id)
        if sp.exists():
            old = json.loads(sp.read_text(encoding="utf-8"))
            if old.get("thresholds_sha256") != th.sha256:
                raise HealthError(
                    f"{sp.name}: başka eşik dosyasıyla yazılmış (sha256 farklı); aynı eşikleri "
                    "kullan ya da bu koşulun özetini ve kapılı sonucunu sil"
                )
            print(f"[{cond.id}] zaten var, atlandı", flush=True)
            continue
        cdir = Path(work) / cond.id
        shutil.rmtree(cdir, ignore_errors=True)  # önceki oturumdan yarım kalmış olabilir
        name = gate_name(model.run_id, split, cond.id)
        try:
            make_condition(
                src, cdir, split, cond, {"rgbt4"}, seed, workers, label_set=LABEL_SET, hook=hook
            )
            text = (cdir / "hook.jsonl").read_text(encoding="utf-8")
            frames = [json.loads(line) for line in text.splitlines()]
            log = Path(project) / name / "health.jsonl.gz"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_bytes(gzip.compress(text.encode("utf-8")))
            summary = summarize_health(frames)
            evaluated = summary["changed"] > 0
            if evaluated:
                evaluate(
                    model.cfg,
                    config=model.config,
                    weights=model.weights,
                    data=cdir / Path(model.cfg.data).name,
                    split=split,
                    meta=meta,
                    project=project,
                    results=results,
                    n_boot=0,
                    device=device,
                    label_sets=(LABEL_SET,),
                    condition={
                        "id": f"{cond.id}_gate",
                        "base": cond.id,
                        "corruption": cond.corruption,
                        "severity": cond.severity,
                        "target": cond.target,
                        "seed": seed,
                        "thresholds_sha256": th.sha256,
                        "gate": summary["gate"],
                    },
                    results_subdir=GATE_DIR,
                    workers=workers,
                )
                # Ultralytics'in ham tahmin kopyası Drive'da yer kaplar; bizimki .json.gz'de
                (Path(project) / name / "predictions.json").unlink(missing_ok=True)
                n += 1
            write_json(
                sp,
                {
                    "run_id": model.run_id,
                    "split": split,
                    "condition": cond.id,
                    "seed": seed,
                    "thresholds_sha256": th.sha256,
                    "evaluated": evaluated,
                    **summary,
                },
            )
            g = summary["gate"]
            print(
                f"[{cond.id}] kapı ir_off={g['ir_off']} rgb_off={g['rgb_off']} "
                f"değişen={summary['changed']}/{summary['n_images']} "
                f"{'değerlendirildi' if evaluated else 'değerlendirme yok (Faz 1 tahmini)'}",
                flush=True,
            )
        finally:
            shutil.rmtree(cdir, ignore_errors=True)
    return n
