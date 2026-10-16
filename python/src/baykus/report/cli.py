"""baykus-report: results/runs/*.json → docs/results.md + grafik."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from baykus.edge.config import load_edge_config
from baykus.report.edge import apply_rule, edge_rows, edge_table, load_edge, rule_text
from baykus.report.plots import illumination_plot
from baykus.report.tables import (
    compare_table,
    label_set_table,
    load_compares,
    load_runs,
    m3ot_table,
    main_table,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="baykus-report")
    p.add_argument("--runs", type=Path, default=Path("../results/runs"))
    p.add_argument("--compare", type=Path, default=Path("../results/compare"))
    p.add_argument("--out", type=Path, default=Path("../docs/results.md"))
    p.add_argument("--fig", type=Path, default=Path("../docs/figures/results/illumination.png"))
    p.add_argument("--split", default="val")
    p.add_argument("--label-set", default="ir")
    p.add_argument("--m3ot", type=Path, default=Path("../results/m3ot"))
    p.add_argument("--m3ot-split", default="test")
    p.add_argument("--edge-results", type=Path, default=Path("../results"))
    p.add_argument("--edge-config", type=Path, default=Path("../configs/edge/e9_edge.yaml"))
    a = p.parse_args(argv)
    runs = load_runs(a.runs, split=a.split, label_set=a.label_set)
    illumination_plot(runs, a.fig)
    rel = Path(os.path.relpath(a.fig, a.out.parent)).as_posix()
    text = (
        f"# Sonuçlar ({a.split}, {a.label_set} etiketleri)\n\n"
        "> Bu dosya `baykus-report` ile `results/runs/*.json`'dan üretildi; elle düzenlenmez.\n"
        "> Köşeli parantez: görüntü düzeyinde bootstrap %95 güven aralığı (FR-V8).\n\n"
        f"{main_table(runs)}\n\n![Aydınlatma kırılımı]({rel})\n\n"
        "## Etiket seti karşılaştırması (E5)\n\n"
        "> Aynı tahminler iki etiket setine karşı (FR-V3). Δ = RGB − IR, mAP50 puanı.\n\n"
        f"{label_set_table(load_runs(a.runs, split=a.split))}\n"
    )
    compares = load_compares(a.compare, split=a.split)
    if compares:
        text += (
            "\n## Eşleştirilmiş farklar\n\n"
            "> `baykus-eval compare`: aynı bootstrap örnekleriyle A − B, mAP50 puanı "
            "[%95 GA]; `*` aralık 0'ı içermiyor (anlamlı). Tek seed: seed varyansı dahil değil.\n\n"
            f"{compare_table(compares)}\n"
        )
    m3 = load_runs(a.m3ot, split=a.m3ot_split) if a.m3ot.is_dir() else []
    if m3:
        text += (
            "\n## Veri setleri arası test (E7, M3OT)\n\n"
            "> DroneVehicle'da eğitilen modeller M3OT test bölümünde, sınıfsız `vehicle` AP "
            "(spec §8.3): rgb RGB etiketiyle, diğer modlar IR etiketiyle. M3OT kutuları çok küçük "
            "(< 1024 px²): mutlak değer değil modlar arası sıra yorumlanır.\n\n"
            f"{m3ot_table(m3)}\n"
        )
    acc, bench = load_edge(a.edge_results)
    if acc or bench:
        cfg = load_edge_config(a.edge_config)
        rows = edge_rows(acc, bench, runs, cfg, split=a.split)
        text += (
            "\n## Edge (E9)\n\n"
            "> `baykus-edge eval` (val, IR etiketleri, bootstrap yok) ve `baykus-bench` (§14: "
            "batch 1, 640×640, 50 ısınma + 500 ölçüm; toplam = ön işleme + çıkarım + son işleme "
            "[+ WBF]). ΔmAP50 aynı pipeline'ın ONNX FP32'sine göre; PyTorch satırı ana "
            "değerlendirme (dikdörtgen girdi, max_det 1000).\n\n"
            f"{edge_table(rows)}\n\n### Sevk kuralı (ADR 0007)\n\n"
            f"{rule_text(apply_rule(rows, cfg), cfg)}\n"
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    print(a.out)
    return 0
