"""baykus-health: Faz 2 sensör sağlığı kapısı ve güven kalibrasyonu.

fit   — kapı eşiklerini train'de bozulmuş görüntülerden seçer (Youden J).
suite   — sağlamlık paketini kapıyla çalıştırır (E3c + kapı; kaldığı yerden sürer).
compare — E3c + kapı − E3c eşleştirilmiş bootstrap, kabul ölçütleri (gate_<split>.json).
calibrate — sevk edilen modelin skorları için izotonik tablo, testte ECE önce / sonra.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from baykus.edge.calib import calibration_stems
from baykus.edge.export import sha256_file
from baykus.eval.cli import load_coco
from baykus.eval.predictions import load_predictions
from baykus.health.calibrate import calibrate
from baykus.health.compare import gate_report
from baykus.health.features import FEATURES
from baykus.health.fit import feature_rows, fit_thresholds
from baykus.health.suite import run_gated_suite
from baykus.robustness.cli import load_model_spec
from baykus.robustness.corruptions import select_conditions, suite_conditions
from baykus.runinfo import git_info, iso, utc_now, write_json


def _cmd_fit(a: argparse.Namespace) -> int:
    root, meta = Path(a.data_root), Path(a.meta)
    illum = meta / "illumination_thresholds.json"
    if not illum.exists():
        raise FileNotFoundError(f"{illum} yok (ADR 0005 t_dark eşiği gerekli)")
    t_dark = float(json.loads(illum.read_text(encoding="utf-8"))["t_dark"])
    stems = calibration_stems(root, "train", a.n, a.seed)
    rows = feature_rows(root, "train", stems, a.seed, a.workers)
    if a.features_out:
        out = Path(a.features_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "condition", *FEATURES])
            w.writeheader()
            w.writerows(rows)
    rec = fit_thresholds(rows, t_dark)
    rec["source"] = {
        "split": "train",
        "n": len(stems),
        "seed": a.seed,
        "stems_sha256": hashlib.sha256("\n".join(stems).encode("utf-8")).hexdigest(),
    }
    rec["git_commit"] = git_info()["git_commit"]
    rec["created_at"] = iso(utc_now())
    path = write_json(Path(a.out), rec)
    print(path)
    for name, s in rec["states"].items():
        t = s["train"]
        print(
            f"{name:11s} {s['feature']:12s} {s['direction']:5s} eşik={s['threshold']:.3f} "
            f"tpr={t['tpr']:.3f} fpr={t['fpr']:.3f}"
        )
    print(f"train temiz karelerde kapı oranı: {rec['train_clean_gate_rate']:.4f}")
    return 0


def _cmd_suite(a: argparse.Namespace) -> int:
    model = load_model_spec(a.model)
    conds = select_conditions(a.conditions) if a.conditions else suite_conditions()
    n = run_gated_suite(
        model,
        conds,
        Path(a.thresholds),
        src=Path(a.data_root),
        work=Path(a.work),
        meta=Path(a.meta),
        split=a.split,
        project=Path(a.project),
        results=Path(a.results),
        seed=a.seed,
        workers=a.workers,
        device=a.device,
    )
    print(f"kapıyla değerlendirilen koşul: {n}")
    return 0


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _cmd_compare(a: argparse.Namespace) -> int:
    gt = load_coco(Path(a.data_root) / "coco" / f"{a.split}_ir.json")
    rep = gate_report(
        run_id=a.run_id,
        split=a.split,
        robust_preds=Path(a.robust_preds),
        gt=gt,
        results=Path(a.results),
        n_boot=a.n_boot,
        seed=a.seed,
        workers=a.workers,
    )
    out = write_json(Path(a.results) / "health" / f"gate_{a.split}.json", rep)
    print(out)
    h = rep["harmful"]
    if h:
        print(f"Z {h['conditions']}: ort. fark {h['diff'] * 100:+.1f} puan, GA {h['ci95']}")
    acc = rep["acceptance"]
    if acc:
        k1, k2, k3 = acc["k1"], acc["k2"], acc["k3"]
        print(f"Kabul 1 {_mark(k1['passed'])} temiz fark {k1['value'] * 100:+.2f} puan (≥ −0,5)")
        print(f"Kabul 2 {_mark(k2['passed'])} Z birleşik GA {k2['ci95']} (alt sınır > 0)")
        print(f"Kabul 3 {_mark(k3['passed'])} temiz yanlış alarm {k3['value'] * 100:.2f}% (≤ %2)")
    return 0


def _cmd_calibrate(a: argparse.Namespace) -> int:
    coco = Path(a.coco_dir)
    vp, tp = Path(a.val_preds), Path(a.test_preds)
    rec = calibrate(
        load_coco(coco / "val_ir.json"),
        load_predictions(vp),
        load_coco(coco / "test_ir.json"),
        load_predictions(tp),
    )
    rec["model"] = a.model
    rec["label_set"] = "ir"
    rec["sources"] = {
        "val_preds": str(vp),
        "test_preds": str(tp),
        "val_sha256": sha256_file(vp),
        "test_sha256": sha256_file(tp),
    }
    rec["git_commit"] = git_info()["git_commit"]
    rec["created_at"] = iso(utc_now())
    print(write_json(Path(a.out), rec))
    t, k4 = rec["test"], rec["k4"]
    print(
        f"test ECE {t['ece_before']:.4f} → {t['ece_after']:.4f} ({t['n_eval']} tespit, skor ≥ 0,25)"
    )
    print(f"test mAP50 {t['map50_before']:.4f} → {t['map50_after']:.4f}")
    print(f"Kabul 4 {_mark(k4['passed'])} |ΔmAP50| {k4['value'] * 100:.3f} puan (≤ 0,1)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-health")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="kapı eşiklerini train'de seç")
    f.add_argument("--data-root", required=True, help="işlenmiş veri kökü (rgb/, ir/)")
    f.add_argument("--meta", required=True, help="illumination_thresholds.json klasörü")
    f.add_argument("--out", default="../results/health/thresholds.json")
    f.add_argument("--features-out", default=None, help="özellik satırları (CSV)")
    f.add_argument("--n", type=int, default=500)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--workers", type=int, default=1)
    f.set_defaults(func=_cmd_fit)
    s = sub.add_parser("suite", help="sağlamlık paketini kapıyla çalıştır")
    s.add_argument("--model", required=True, help="CONFIG=AĞIRLIK (4 kanallı model, E3c)")
    s.add_argument("--thresholds", required=True, help="baykus-health fit çıktısı")
    s.add_argument("--data-root", required=True)
    s.add_argument("--meta", required=True, help="slices_<split>.json klasörü")
    s.add_argument("--split", default="val", choices=["val", "test"])
    s.add_argument(
        "--conditions", nargs="+", default=None, help="koşul kimlikleri (varsayılan: hepsi)"
    )
    s.add_argument("--work", default="runs/health_work", help="geçici koşul klasörleri")
    s.add_argument("--project", default="runs/robust", help="tahmin ve sağlık kayıtları")
    s.add_argument("--results", default="../results")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--device", default=None)
    s.set_defaults(func=_cmd_suite)
    c = sub.add_parser("compare", help="E3c + kapı − E3c ve kabul ölçütleri")
    c.add_argument("--run-id", required=True)
    c.add_argument("--split", default="val", choices=["val", "test"])
    c.add_argument("--robust-preds", required=True, help="Faz 1 ve kapılı tahmin klasörleri")
    c.add_argument("--data-root", required=True, help="coco/<split>_ir.json kökü")
    c.add_argument("--results", default="../results")
    c.add_argument("--n-boot", type=int, default=1000)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--workers", type=int, default=1)
    c.set_defaults(func=_cmd_compare)
    k = sub.add_parser("calibrate", help="izotonik skor kalibrasyonu ve ECE")
    k.add_argument("--val-preds", required=True, help="val tahminleri (.json.gz, conf ≥ 0,001)")
    k.add_argument("--test-preds", required=True)
    k.add_argument("--coco-dir", required=True, help="val_ir.json ve test_ir.json klasörü")
    k.add_argument("--out", default="../results/health/calibration_e3c_onnx_fp32.json")
    k.add_argument("--model", default="e3c_modality_dropout · onnx_fp32")
    k.set_defaults(func=_cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
