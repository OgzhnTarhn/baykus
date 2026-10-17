"""baykus-health: Faz 2 sensör sağlığı kapısı ve güven kalibrasyonu.

fit — kapı eşiklerini train'de bozulmuş görüntülerden seçer (Youden J).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from baykus.edge.calib import calibration_stems
from baykus.health.features import FEATURES
from baykus.health.fit import feature_rows, fit_thresholds
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
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
