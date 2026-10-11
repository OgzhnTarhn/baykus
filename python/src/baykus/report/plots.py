"""Aydınlatma kırılımı çubuk grafiği (güven aralığı hata çubuğu ile)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from baykus.report.tables import ILLUMINATION  # noqa: E402

LABELS = {"day": "Gündüz", "night": "Gece", "dark_night": "Karanlık gece"}


def illumination_plot(runs: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.8 / max(len(runs), 1)
    x = np.arange(len(ILLUMINATION))
    for i, r in enumerate(runs):
        m = r["metrics"]
        ill = m.get("per_illumination") or {}
        ci = (m.get("ci95") or {}).get("per_illumination") or {}
        vals, err = [], [[], []]
        for k in ILLUMINATION:
            v = ill.get(k)
            lo, hi = ci.get(k) or [None, None]
            vals.append(np.nan if v is None else 100 * v)
            err[0].append(0 if v is None or lo is None else 100 * (v - lo))
            err[1].append(0 if v is None or hi is None else 100 * (hi - v))
        ax.bar(
            x + i * width,
            vals,
            width,
            yerr=err,
            capsize=3,
            label=f"{r['model']['mode']} ({r['label_set']})",
        )
    ax.set_xticks(x + width * (len(runs) - 1) / 2, [LABELS[k] for k in ILLUMINATION])
    ax.set_ylabel("mAP50 (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
