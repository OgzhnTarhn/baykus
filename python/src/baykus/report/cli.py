"""baykus-report: results/runs/*.json → docs/results.md + grafik."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from baykus.report.plots import illumination_plot
from baykus.report.tables import (
    compare_table,
    label_set_table,
    load_compares,
    load_runs,
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
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    print(a.out)
    return 0
