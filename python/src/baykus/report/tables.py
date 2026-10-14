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


LABEL_SET_HEADER = (
    "| Çalıştırma | Mod | mAP50 IR [%95 GA] | mAP50 RGB [%95 GA] | Δ (RGB − IR) | Δ gündüz "
    "| Δ gece | Δ karanlık gece |\n|---|---|---|---|---|---|---|---|"
)
COMPARE_HEADER = (
    "| Karşılaştırma (A − B) | Genel | Gündüz | Gece | Karanlık gece |\n|---|---|---|---|---|"
)


def _delta(a: float | None, b: float | None) -> str:
    return "—" if a is None or b is None else f"{100 * (b - a):+.1f}"


def label_set_table(runs: list[dict]) -> str:
    """E5: aynı tahminler iki etiket setinde (FR-V3). Δ = RGB − IR, puan."""
    pairs: dict[tuple[str, str], dict[str, dict]] = {}
    for r in runs:
        pairs.setdefault((r["run_id"], r["split"]), {})[r["label_set"]] = r
    rows = [LABEL_SET_HEADER]
    for (run_id, _), p in sorted(pairs.items()):
        if "ir" not in p or "rgb" not in p:
            continue
        a, b = p["ir"]["metrics"], p["rgb"]["metrics"]
        ia, ib = a.get("per_illumination") or {}, b.get("per_illumination") or {}
        cells = [
            run_id,
            p["ir"]["model"]["mode"],
            fmt(a.get("map50"), (a.get("ci95") or {}).get("map50")),
            fmt(b.get("map50"), (b.get("ci95") or {}).get("map50")),
            _delta(a.get("map50"), b.get("map50")),
            *(_delta(ia.get(k), ib.get(k)) for k in ILLUMINATION),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _diff(d: dict | None) -> str:
    if not d or d.get("diff") is None:
        return "—"
    lo, hi = d["ci95"]
    s = f"{100 * d['diff']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return s + (" *" if d.get("significant") else "")


def compare_table(compares: list[dict]) -> str:
    """FR-V8 eşleştirilmiş bootstrap; fark = A − B (puan), * = aralık 0'ı içermiyor."""
    rows = [COMPARE_HEADER]
    for c in sorted(compares, key=lambda c: c["name"]):
        ill = c.get("per_illumination") or {}
        cells = [c["name"], _diff(c.get("overall")), *(_diff(ill.get(k)) for k in ILLUMINATION)]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def load_compares(folder: Path) -> list[dict]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
