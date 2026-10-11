"""baykus-report: results/runs/*.json → docs/results.md + grafik."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from baykus.report.plots import illumination_plot
from baykus.report.tables import load_runs, main_table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="baykus-report")
    p.add_argument("--runs", type=Path, default=Path("../results/runs"))
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
        f"{main_table(runs)}\n\n![Aydınlatma kırılımı]({rel})\n"
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    print(a.out)
    return 0
