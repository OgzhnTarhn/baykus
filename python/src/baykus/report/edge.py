"""Edge tablosu ve sevk kuralı (H5, spec §14, ADR 0007): results/edge + results/bench → tablo.

ΔmAP50 aynı pipeline'ın onnx_fp32'sine göre (H5 planı Spec yorumları 2); PyTorch satırı ayrıca.
Kural yalnız laptop CPU satırlarıyla, config'teki aday sırasıyla uygulanır.
"""

from __future__ import annotations

import json
from pathlib import Path

from baykus.edge.config import EdgeConfig, Pipeline
from baykus.edge.variants import VARIANTS
from baykus.report.tables import fmt

RULE_HARDWARE = "laptop-cpu"
HEADER = (
    "| Pipeline | Donanım | Format | Hassasiyet | Boyut (MB) | mAP50 | ΔmAP50 (puan) | p50 (ms) "
    "| p95 (ms) | FPS |\n|---|---|---|---|---|---|---|---|---|---|"
)


def load_edge(results: Path) -> tuple[list[dict], list[dict]]:
    def load(folder: Path) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))
                if not p.name.startswith("parity_")]  # fmt: skip

    return load(Path(results) / "edge"), load(Path(results) / "bench")


def _experiment(selection: str) -> str:
    return Path(selection).stem.removesuffix("_val_selection")


def _accuracy(recs: list[dict], pipe: Pipeline, cfg: EdgeConfig, split: str) -> dict:
    """variant (None = PyTorch / ana değerlendirme) → kayıt; yalnız IR etiketi ve split."""
    out: dict = {}
    for r in recs:
        if r.get("split") != split or r.get("label_set") != "ir":
            continue
        f = r.get("fusion")
        if pipe.kind == "single":
            if f is None and r["run_id"] == cfg.models[pipe.model].run_id:
                out[(r.get("edge") or {}).get("variant")] = r
        elif f and (f.get("selection"), f.get("rgb_run_id"), f.get("ir_run_id")) == (
            _experiment(pipe.selection), cfg.models[pipe.rgb].run_id, cfg.models[pipe.ir].run_id
        ):  # fmt: skip
            out[f.get("variant")] = r
    return out


def _size(
    acc: list[dict], pipe: Pipeline, cfg: EdgeConfig, variant: str, split: str
) -> float | None:
    names = [pipe.model] if pipe.kind == "single" else [pipe.rgb, pipe.ir]
    sizes = []
    for n in names:
        rec = _accuracy(acc, Pipeline("single", model=n), cfg, split).get(variant)
        if rec is None:
            return None
        sizes.append(rec["edge"]["file_size_mb"])
    return float(sum(sizes))


def _map50(rec: dict | None) -> float | None:
    return None if rec is None else rec["metrics"]["map50"]


def edge_rows(
    acc: list[dict], bench: list[dict], runs: list[dict], cfg: EdgeConfig, split: str = "val"
) -> list[dict]:
    rows: list[dict] = []
    for name, pipe in cfg.pipelines.items():
        by_var = _accuracy(acc, pipe, cfg, split)
        ref = _accuracy(runs, pipe, cfg, split).get(None)
        rows.append({"pipeline": name, "variant": "pytorch", "hardware": None,
                     "format": "pytorch", "precision": "fp32", "size_mb": None,
                     "map50": _map50(ref), "delta_map50": None, "p50": None, "p95": None,
                     "fps": None})  # fmt: skip
        fp32 = _map50(by_var.get("onnx_fp32"))
        benches = [b for b in bench if b["pipeline"] == name]
        seen = {v for v in by_var if v} | {b["variant"] for b in benches}
        for v in (v for v in VARIANTS if v in seen):
            m = _map50(by_var.get(v))
            hws = [b for b in benches if b["variant"] == v] or [None]
            for b in sorted(hws, key=lambda x: "" if x is None else x["hardware"]):
                t = (b or {}).get("stages", {}).get("total", {})
                rows.append({
                    "pipeline": name, "variant": v, "hardware": b and b["hardware"],
                    "format": VARIANTS[v].fmt, "precision": VARIANTS[v].precision,
                    "size_mb": _size(acc, pipe, cfg, v, split), "map50": m,
                    "delta_map50": None if m is None or fp32 is None else m - fp32,
                    "p50": t.get("p50"), "p95": t.get("p95"), "fps": (b or {}).get("fps"),
                })  # fmt: skip
    return rows


def _num(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def edge_table(rows: list[dict]) -> str:
    out = [HEADER]
    for r in rows:
        d = r["delta_map50"]
        cells = [
            r["pipeline"],
            r["hardware"] or "—",
            r["format"],
            r["precision"]
            + (
                " (head FP32)"
                if VARIANTS.get(r["variant"], None) and VARIANTS[r["variant"]].head_fp32
                else ""
            ),  # fmt: skip
            _num(r["size_mb"]),
            fmt(r["map50"]),
            "—" if d is None else f"{100 * d:+.1f}",
            _num(r["p50"]),
            _num(r["p95"]),
            _num(r["fps"]),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def apply_rule(rows: list[dict], cfg: EdgeConfig) -> dict:
    trace, shipped = [], None
    for name in cfg.rule.order:
        eligible = [
            r for r in rows
            if r["pipeline"] == name and r["hardware"] == RULE_HARDWARE
            and r["delta_map50"] is not None and r["p95"] is not None
            and round(r["delta_map50"], 6) >= -cfg.rule.max_map50_drop  # tam −1,0 puan geçer
        ]  # fmt: skip
        best = min(eligible, key=lambda r: r["p95"], default=None)
        passes = best is not None and best["p95"] <= cfg.rule.target_p95_ms
        trace.append({"pipeline": name, "eligible": [r["variant"] for r in eligible],
                      "best": best and best["variant"], "p95": best and best["p95"],
                      "passes": passes})  # fmt: skip
        if passes and shipped is None:
            shipped = {k: best[k] for k in ("pipeline", "variant", "p95", "map50", "delta_map50")}
    return {"shipped": shipped, "trace": trace}


def rule_text(result: dict, cfg: EdgeConfig) -> str:
    lines = [
        f"> Hedef: laptop CPU, toplam p95 ≤ {cfg.rule.target_p95_ms:g} ms; varyant ONNX FP32'den "
        f"en fazla {100 * cfg.rule.max_map50_drop:.1f} puan düşük (ADR 0007).",
        "",
        "| Aday | Uygun varyantlar | En hızlı | p95 (ms) | Geçti mi |",
        "|---|---|---|---|---|",
    ]
    for t in result["trace"]:
        lines.append(f"| {t['pipeline']} | {', '.join(t['eligible']) or '—'} | "
                     f"{t['best'] or '—'} | {_num(t['p95'])} | "
                     f"{'evet' if t['passes'] else 'hayır'} |")  # fmt: skip
    s = result["shipped"]
    lines += ["", f"**Sevk:** {s['pipeline']} · {s['variant']}" if s else
              "**Sevk:** hiçbir aday hedefi tutmadı → E8-n (ADR 0007 madde 6)"]  # fmt: skip
    return "\n".join(lines)
