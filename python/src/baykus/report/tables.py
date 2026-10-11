"""results/runs/*.json → markdown tablo. Eksik değer '—'; tahmini değer yazılmaz."""

from __future__ import annotations

import json
from pathlib import Path

ILLUMINATION = ("day", "night", "dark_night")
HEADER = (
    "| Çalıştırma | Mod | Split | Etiket | mAP50 [%95 GA] | mAP50-95 | Gündüz | Gece "
    "| Karanlık gece |\n|---|---|---|---|---|---|---|---|---|"
)


def load_runs(folder: Path, split: str | None = None, label_set: str | None = None) -> list[dict]:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(folder).glob("*.json"))]
    runs = [
        r
        for r in runs
        if (split is None or r.get("split") == split)
        and (label_set is None or r.get("label_set") == label_set)
    ]
    return sorted(runs, key=lambda r: r["run_id"])


def fmt(v: float | None, ci: list | None = None) -> str:
    if v is None:
        return "—"
    s = f"{100 * v:.1f}"
    if ci and None not in ci:
        s += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
    return s


def main_table(runs: list[dict]) -> str:
    rows = [HEADER]
    for r in runs:
        m = r["metrics"]
        ill = m.get("per_illumination") or {}
        cells = [
            r["run_id"],
            r["model"]["mode"],
            r["split"],
            r["label_set"],
            fmt(m.get("map50"), (m.get("ci95") or {}).get("map50")),
            fmt(m.get("map50_95")),
            *(fmt(ill.get(k)) for k in ILLUMINATION),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
