"""baykus-health: Faz 2 sensör sağlığı kapısı ve güven kalibrasyonu.

fit   — kapı eşiklerini train'de bozulmuş görüntülerden seçer (Youden J).
suite   — sağlamlık paketini kapıyla çalıştırır (E3c + kapı; kaldığı yerden sürer).
compare — E3c + kapı − E3c eşleştirilmiş bootstrap, kabul ölçütleri (gate_<split>.json).
calibrate — sevk edilen modelin skorları için izotonik tablo, testte ECE önce / sonra.
report  — results/health/*.json → docs/health.md + güvenilirlik diyagramı.
guard   — Faz 3 kare kuralı: train'de r_min seçimi (gate_model.json).
guard-files — val keşfi için Drive'dan inecek dosyaların listesi.
guard-val — val keşfi: Faz 2 kapılı ve taban tahminlerinin kare kuralıyla birleştirilmesi.
m3ot    — M3OT dizilerinde sağlık protokolü (V: geliştirme bölümü, demo dizisi; T: yalnız rapor).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

from baykus.edge.calib import calibration_stems
from baykus.edge.export import sha256_file
from baykus.eval.cli import load_coco
from baykus.eval.predictions import load_predictions
from baykus.health.calibrate import calibrate
from baykus.health.compare import check_thresholds, gate_report
from baykus.health.features import FEATURES
from baykus.health.fit import feature_rows, fit_thresholds
from baykus.health.gate import load_thresholds
from baykus.health.guard import (
    download_list,
    guard_val_report,
    load_gate_model,
    select_r_min,
    write_gate_model,
)
from baykus.health.report import calibration_plot, health_markdown
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
    th = Path(a.thresholds) if a.thresholds else Path(a.results) / "health" / "thresholds.json"
    if not th.exists():
        raise FileNotFoundError(f"{th} yok: özetler hangi eşiklerle üretildi denetlenemez")
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
        thresholds=th,
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
        alarm = "—" if k3["value"] is None else f"{k3['value'] * 100:.2f}%"
        print(f"Kabul 3 {_mark(k3['passed'])} temiz yanlış alarm {alarm} (≤ %2)")
    return 0


def _num(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


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
        f"test ECE {_num(t['ece_before'])} → {_num(t['ece_after'])} "
        f"({t['n_eval']} tespit, skor ≥ 0,25)"
    )
    print(f"test mAP50 {t['map50_before']:.4f} → {t['map50_after']:.4f}")
    print(f"Kabul 4 {_mark(k4['passed'])} |ΔmAP50| {k4['value'] * 100:.3f} puan (≤ 0,1)")
    return 0


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _cmd_report(a: argparse.Namespace) -> int:
    h, out, fig = Path(a.health), Path(a.out), Path(a.fig)
    th = _read(h / "thresholds.json")
    if th is None:
        raise FileNotFoundError(f"{h / 'thresholds.json'} yok (önce baykus-health fit)")
    calib = _read(h / "calibration_e3c_onnx_fp32.json")
    gate_val, gate_test = _read(h / "gate_val.json"), _read(h / "gate_test.json")
    sha = sha256_file(h / "thresholds.json")
    check_thresholds(sha, {"gate_val.json": gate_val, "gate_test.json": gate_test})
    rel = Path(os.path.relpath(fig, out.parent)).as_posix()
    text = health_markdown(th, gate_val, gate_test, calib, rel)
    if calib:
        calibration_plot(calib, fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)
    return 0


E3C_RUN = "2026-10-15_e3c-modality-dropout_s_d4e1280"  # sevk edilen model (ADR 0007)


def _pct2(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _cmd_guard(a: argparse.Namespace) -> int:
    th = load_thresholds(Path(a.thresholds))
    with Path(a.features).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    sel = select_r_min(rows, th)
    print(write_gate_model(Path(a.out), sel, th, Path(a.features)))
    t = sel["train_tpr"]
    print(
        f"r_min={sel['r_min']:.4f} ({sel['criterion']}) · train temiz alarm "
        f"{_pct2(sel['train_clean_alarm_rate'])} ({sel['n_clean']} kare)"
    )
    print(
        f"train misalign-20/40 yakalama: tümü {_pct2(t['all'])}, karanlık {_pct2(t['dark'])}, "
        f"aydınlık {_pct2(t['bright'])} ({sel['n_pos']} kare)"
    )
    return 0


def _cmd_guard_files(a: argparse.Namespace) -> int:
    for path in download_list(Path(a.results), a.run_id):
        print(path)
    return 0


def _cmd_guard_val(a: argparse.Namespace) -> int:
    th = load_thresholds(Path(a.thresholds))
    rep = guard_val_report(
        run_id=a.run_id,
        robust_preds=Path(a.robust_preds),
        gt=load_coco(Path(a.gt)),
        results=Path(a.results),
        gate_model=load_gate_model(Path(a.gate_model)),
        th=th,
        n_boot=a.n_boot,
        seed=a.seed,
        workers=a.workers,
    )
    rep["git_commit"] = git_info()["git_commit"]
    rep["created_at"] = iso(utc_now())
    print(write_json(Path(a.out), rep))
    acc = rep["acceptance"]
    if acc:
        k1, k2, k3 = acc["k1"], acc["k2"], acc["k3"]
        print(f"({rep['label']}) r_min={rep['r_min']:.4f}")
        print(f"Kabul 1 {_mark(k1['passed'])} temiz fark {k1['value'] * 100:+.2f} puan (≥ −0,5)")
        print(f"Kabul 2 {_mark(k2['passed'])} Z birleşik GA {k2['ci95']} (alt sınır > 0)")
        print(f"Kabul 3 {_mark(k3['passed'])} temiz yanlış alarm {_pct2(k3['value'])} (≤ %2)")
    else:
        print("kabul ölçütleri hesaplanamadı (temiz ya da Z koşulu birleştirilemedi)")
    return 0


TEST_FLAG = "--i-know-this-is-test"


def _cmd_m3ot(a: argparse.Namespace) -> int:
    from baykus.engine_ref.models import load_models
    from baykus.health.m3ot import CONDITIONS, RULES, combine, pick_demo, run_sequence

    if a.test and not a.i_know_this_is_test:
        raise ValueError(f"M3OT test yalnız son ölçümde {TEST_FLAG} ile ölçülür")
    if a.test and a.pick_demo:
        raise ValueError("demo dizisi test'te seçilmez (--pick-demo yalnız --dev)")
    cfg_path = Path(a.config)
    if a.dev:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        data = Path(a.data or "../data/m3ot_dev")
        split, seqs = cfg["split"], list(cfg["sequences"])
    else:
        data = Path(a.data or "../data/m3ot_test")
        split, seqs = "test", sorted(p.name for p in (data / "sequences").iterdir())
    models = load_models(Path(a.models), Path(a.manifest) if a.manifest else None)
    results = []
    for s in seqs:
        r = run_sequence(data / "sequences" / s, data / "labels", models, keep_coco=True)
        if r["split"] != split:
            raise ValueError(f"{s}: index.json split {r['split']!r}, beklenen {split!r}")
        results.append(r)
        t = r["rules"]["frame+temporal"]
        print(
            f"[{s}] kare+zaman: temiz hizasız {t['clean']['alarm_rates']['misaligned']:.4f} · "
            f"misalign-20 {t['misalign-20']['alarm_rates']['misaligned']:.4f} · "
            f"gecikme {t['misalign-20@K']['delay']} · demo ölçütü {_mark(r['demo']['passed'])}",
            flush=True,
        )
    rec = {
        "protocol": "V" if a.dev else "T",
        "split": split,
        "label_set": "ir",
        "class_agnostic": True,
        "rules": list(RULES),
        "conditions": list(CONDITIONS),
        "r_min": models.gate.r_min,
        "gate_model_sha256": models.gate.sha256,
        "thresholds_sha256": models.thresholds.sha256,
        "combined": combine(results, data / "labels", split),
        "sequences": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
    }
    if a.pick_demo:
        rec["demo_choice"] = pick_demo(results, cfg_path)
        print(f"demo dizisi: {rec['demo_choice']['demo']} (ölçüt {rec['demo_choice']})")
    rec["git_commit"] = git_info()["git_commit"]
    rec["created_at"] = iso(utc_now())
    print(write_json(Path(a.out), rec))
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
    c.add_argument(
        "--thresholds", default=None, help="varsayılan: <results>/health/thresholds.json"
    )
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
    r = sub.add_parser("report", help="docs/health.md ve güvenilirlik diyagramı")
    r.add_argument("--health", default="../results/health")
    r.add_argument("--out", default="../docs/health.md")
    r.add_argument("--fig", default="../docs/figures/results/calibration.png")
    r.set_defaults(func=_cmd_report)
    g = sub.add_parser("guard", help="kare kuralı: train'de r_min seç (gate_model.json)")
    g.add_argument("--features", default="../data/health/train_features.csv")
    g.add_argument("--thresholds", default="../results/health/thresholds.json")
    g.add_argument("--out", default="../results/health/gate_model.json")
    g.set_defaults(func=_cmd_guard)
    gf = sub.add_parser("guard-files", help="val keşfi için Drive'dan inecek dosyalar")
    gf.add_argument("--results", default="../results")
    gf.add_argument("--run-id", default=E3C_RUN)
    gf.set_defaults(func=_cmd_guard_files)
    gv = sub.add_parser("guard-val", help="val keşfi: kare kuralıyla birleştirilmiş kapı")
    gv.add_argument("--gate-model", default="../results/health/gate_model.json")
    gv.add_argument("--robust-preds", required=True, help="Faz 1 ve kapılı tahmin klasörleri")
    gv.add_argument("--gt", required=True, help="val_ir.json (Faz 2 compare'deki COCO)")
    gv.add_argument("--thresholds", default="../results/health/thresholds.json")
    gv.add_argument("--results", default="../results")
    gv.add_argument("--run-id", default=E3C_RUN)
    gv.add_argument("--out", default="../results/health/gate_model_val.json")
    gv.add_argument("--n-boot", type=int, default=1000)
    gv.add_argument("--seed", type=int, default=42)
    gv.add_argument("--workers", type=int, default=1)
    gv.set_defaults(func=_cmd_guard_val)
    m = sub.add_parser("m3ot", help="M3OT sağlık protokolü (dizi başına, üç kural)")
    which = m.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--dev", action="store_true", help="geliştirme bölümü (configs/m3ot_dev.json)"
    )
    which.add_argument("--test", action="store_true", help="M3OT test (yalnız son ölçüm)")
    m.add_argument(TEST_FLAG, action="store_true")
    m.add_argument("--models", required=True, help="model dizini")
    m.add_argument("--manifest", default=None, help="configs/engine/models.json")
    m.add_argument("--config", default="../configs/m3ot_dev.json")
    m.add_argument("--data", default=None, help="paket kökü (sequences/, labels/)")
    m.add_argument("--out", required=True)
    m.add_argument("--pick-demo", action="store_true", help="demo dizisini config'e yaz (--dev)")
    m.set_defaults(func=_cmd_m3ot)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows konsolu (cp1254) ✅ / ≤ / → yazamaz: kodlanamayan karakter '?' olur, komut çökmez
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
