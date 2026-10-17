"""F2 sonrası kontroller (2026-10-17, dal incelemesinin önerileri; yeniden Colab gerekmez).

1. Temiz val'de kapının yanlış alarmları: neden, aydınlatma (ADR 0005 eşikleri), faz korelasyonu
   yanıtı ve ölçülen kayma.
2. Eşiklerin tam sayı Youden J ile yeniden hesabı (train_features.csv; Colab'dakiyle aynı mı).
3. Kapının dokunmadığı karelerde kapılı koşunun tahminleri Faz 1 tahminleriyle aynı mı.
4. Kalibrasyon adım fonksiyonunun skor eşitlikleri; eşitlik ham skorla bozulunca test mAP50.

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

from baykus.eval.cli import load_coco
from baykus.eval.coco_eval import evaluate_coco
from baykus.eval.predictions import load_predictions
from baykus.health.calibrate import apply_isotonic
from baykus.health.features import FEATURES
from baykus.health.fit import fit_thresholds

ROOT = Path(__file__).resolve().parents[2]
RUN = "2026-10-15_e3c-modality-dropout_s_d4e1280"
HEALTH = ROOT / "data/health"
GATED = HEALTH / f"{RUN}_val_clean_gate"
T_DARK, T_DAY = 48.655, 114.14  # ADR 0005 (Colab oturumunun illum-relabel çıktısı)


def _jsonl_gz(path: Path) -> list[dict]:
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _illumination(brightness: float) -> str:
    return "dark_night" if brightness < T_DARK else "night" if brightness < T_DAY else "day"


def false_alarms(frames: list[dict]) -> None:
    fired = [f for f in frames if f["gate"] != "none"]
    print(
        f"temiz val: {len(frames)} kare, kapı {len(fired)}",
        dict(Counter(f["reason"] for f in fired)),
    )
    print(
        "aydınlatma (hepsi):", dict(Counter(_illumination(f["rgb"]["brightness"]) for f in frames))
    )
    print(
        "aydınlatma (alarm):", dict(Counter(_illumination(f["rgb"]["brightness"]) for f in fired))
    )

    def median(rows: list[dict], key: str) -> float:
        return statistics.median(r["pair"][key] for r in rows if r["pair"][key] is not None)

    quiet = [f for f in frames if f["gate"] == "none"]
    print(f"faz korelasyonu yanıtı medyanı: alarm {median(fired, 'response'):.3f}, "
          f"alarm yok {median(quiet, 'response'):.3f}")  # fmt: skip
    print(f"ölçülen kayma medyanı: alarm {median(fired, 'misalign_px'):.1f} px")


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
    print(f"kapının dokunmadığı {len(keep)} karenin {same}'inde tahminler aynı")


def calibration_ties(calibration: Path, test_preds: Path) -> None:
    cal = json.loads(calibration.read_text(encoding="utf-8"))
    x, y = np.array(cal["table"]["x"]), np.array(cal["table"]["y"])
    dets = load_predictions(test_preds)
    raw = np.array([d["score"] for d in dets])
    new = apply_isotonic(raw, x, y)
    print(f"skorlar: ham {len(np.unique(raw))} farklı değer → kalibre {len(np.unique(new))}")
    tie_broken = [
        {**d, "score": float(s + 1e-9 * r)} for d, s, r in zip(dets, new, raw, strict=True)
    ]
    gt = load_coco(ROOT / "data/dv_rgbt/coco/test_ir.json")
    m = evaluate_coco(gt, tie_broken)["map50"]
    t = cal["test"]
    print(f"test mAP50: önce {t['map50_before']:.6f}, sonra {t['map50_after']:.6f}, "
          f"eşitlik ham skorla bozulunca {m:.6f}")  # fmt: skip


if __name__ == "__main__":
    frames = _jsonl_gz(GATED / "health.jsonl.gz")
    false_alarms(frames)
    integer_j(HEALTH / "train_features.csv", ROOT / "results/health/thresholds.json")
    untouched_frames(
        frames,
        GATED / "predictions_coco.json.gz",
        ROOT / f"data/gap/{RUN}_val/predictions_coco.json.gz",
    )
    calibration_ties(
        ROOT / "results/health/calibration_e3c_onnx_fp32.json",
        HEALTH / f"{RUN}_test_onnx_fp32/predictions_coco.json.gz",
    )
