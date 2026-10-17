"""F2 sonrası kontroller (2026-10-17, dal incelemesinin önerileri; yeniden Colab gerekmez).

1. Temiz val'de kapının yanlış alarmları: neden, aydınlatma (ADR 0005 eşikleri), faz korelasyonu
   yanıtı ve ölçülen kayma.
2. Eşiklerin tam sayı Youden J ile yeniden hesabı (train_features.csv; Colab'dakiyle aynı mı).
3. Kapının dokunmadığı karelerde kapılı koşunun tahminleri Faz 1'in ana değerlendirme
   tahminleriyle (data/gap/<run>_val, baykus-eval) aynı mı. `compare`'in eşleştirdiği sağlamlık
   tabanı (Drive robust/<run>_val_clean) laptopta yok; o dosyayla ayrıca karşılaştırılmadı.
4. Kalibrasyon adım fonksiyonunun skor eşitlikleri, val'de (CLAUDE.md: test yalnız final
   ölçümde): tablo val'den, val'e uygulanır; eşitlik ham skorla bozulunca mAP50.

Çalıştırma (python/ içinden, dosyalar data/ altında):
  uv run python ../scripts/analysis/f2_checks.py
Çıktı docs/experiments.md "F2 · kontroller" girdisine yazılır.
"""

from __future__ import annotations

import csv
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from baykus.data.illumination import DAY_MIN_BRIGHTNESS, assign
from baykus.eval.cli import load_coco
from baykus.eval.coco_eval import evaluate_coco
from baykus.eval.predictions import load_predictions
from baykus.health.calibrate import apply_isotonic, fit_isotonic, match_labels
from baykus.health.features import FEATURES
from baykus.health.fit import fit_thresholds

ROOT = Path(__file__).resolve().parents[2]
RUN = "2026-10-15_e3c-modality-dropout_s_d4e1280"
HEALTH = ROOT / "data/health"
GATED = HEALTH / f"{RUN}_val_clean_gate"


def _jsonl_gz(path: Path) -> list[dict]:
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines()]


def false_alarms(frames: list[dict], dark: float) -> None:
    """dark: ADR 0005 t_dark (thresholds.json'daki dark_brightness); gündüz sınırı sabit."""
    levels = assign(np.array([f["rgb"]["brightness"] for f in frames]), (dark, DAY_MIN_BRIGHTNESS))
    fired = [f for f in frames if f["gate"] != "none"]
    fired_levels = [lv for f, lv in zip(frames, levels, strict=True) if f["gate"] != "none"]
    print(
        f"temiz val: {len(frames)} kare, kapı {len(fired)}",
        dict(Counter(f["reason"] for f in fired)),
    )
    print("aydınlatma (hepsi):", dict(Counter(levels)))
    print("aydınlatma (alarm):", dict(Counter(fired_levels)))

    def median(rows: list[dict], key: str) -> str:
        vals = [r["pair"][key] for r in rows if r["pair"][key] is not None]
        return f"{statistics.median(vals):.3f}" if vals else "—"  # alarm yoksa medyan yok

    quiet = [f for f in frames if f["gate"] == "none"]
    print(f"faz korelasyonu yanıtı medyanı: alarm {median(fired, 'response')}, "
          f"alarm yok {median(quiet, 'response')}")  # fmt: skip
    print(f"ölçülen kayma medyanı: alarm {median(fired, 'misalign_px')} px")


def integer_j(features_csv: Path, thresholds: Path) -> None:
    rows = []
    with features_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vals = {k: float(r[k]) if r[k] not in ("", "None") else None for k in FEATURES}
            rows.append({"stem": r["stem"], "condition": r["condition"], **vals})
    old = json.loads(thresholds.read_text(encoding="utf-8"))
    new = fit_thresholds(rows, old["dark_brightness"])
    for name, s in old["states"].items():
        same = s["threshold"] == new["states"][name]["threshold"]
        print(f"eşik {name}: {s['threshold']:.6f} → {new['states'][name]['threshold']:.6f}",
              "aynı" if same else "FARKLI")  # fmt: skip


def untouched_frames(frames: list[dict], gated: Path, base: Path) -> None:
    gt = load_coco(ROOT / "data/dv_rgbt/coco/val_ir.json")
    ids = {Path(i["file_name"]).stem: i["id"] for i in gt.dataset["images"]}
    keep = {ids[f["stem"]] for f in frames if f["gate"] == "none"}

    def per_image(path: Path) -> dict[int, list]:
        out = defaultdict(list)
        for d in load_predictions(path):
            if d["image_id"] in keep:
                box = tuple(round(v, 2) for v in d["bbox"])
                out[d["image_id"]].append((d["category_id"], box, round(d["score"], 4)))
        return {k: sorted(v) for k, v in out.items()}

    a, b = per_image(gated), per_image(base)
    same = sum(a.get(i, []) == b.get(i, []) for i in keep)
    print(
        f"kapının dokunmadığı {len(keep)} karenin {same}'inde tahminler ana değerlendirmeyle aynı"
    )


def calibration_ties(val_preds: Path) -> None:
    gt = load_coco(ROOT / "data/dv_rgbt/coco/val_ir.json")
    dets = load_predictions(val_preds)
    x, y = fit_isotonic(*match_labels(gt, dets))
    raw = np.array([d["score"] for d in dets])
    new = apply_isotonic(raw, x, y)
    print(f"val skorları: ham {len(np.unique(raw))} farklı değer → kalibre {len(np.unique(new))}")

    def with_scores(scores: np.ndarray) -> list[dict]:
        return [{**d, "score": float(s)} for d, s in zip(dets, scores, strict=True)]

    before = evaluate_coco(gt, dets)["map50"]
    after = evaluate_coco(gt, with_scores(new))["map50"]
    tied = evaluate_coco(gt, with_scores(new + 1e-9 * raw))["map50"]
    print(f"val mAP50: önce {before:.6f}, sonra {after:.6f}, "
          f"eşitlik ham skorla bozulunca {tied:.6f}")  # fmt: skip


if __name__ == "__main__":
    frames = _jsonl_gz(GATED / "health.jsonl.gz")
    thresholds = ROOT / "results/health/thresholds.json"
    false_alarms(frames, json.loads(thresholds.read_text(encoding="utf-8"))["dark_brightness"])
    integer_j(HEALTH / "train_features.csv", thresholds)
    untouched_frames(
        frames,
        GATED / "predictions_coco.json.gz",
        ROOT / f"data/gap/{RUN}_val/predictions_coco.json.gz",
    )
    calibration_ties(ROOT / f"data/gap/{RUN}_val_onnx_fp32/predictions_coco.json.gz")
