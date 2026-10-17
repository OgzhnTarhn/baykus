"""E3c + kapı − E3c: eşleştirilmiş bootstrap ve kabul ölçütleri (Faz 2 tasarımı §6.1-6.2, §6.4).

Taban kol Faz 1'in sağlamlık tahminleri (<run>_<split>_<koşul>), kapılı kol aynı klasörün
_gate eki. Kapı pikselleri değiştirmediyse kapılı kol = taban (fark tam 0, bootstrap yok).
Kabul ölçütleri yalnız val'de; test kontrolünde seçim yapılmaz (§6.4).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pycocotools.coco import COCO

from baykus.eval.bootstrap import ap50, collect_matches, paired_diff_ci, paired_mean_diff_ci
from baykus.eval.cli import run_name
from baykus.eval.predictions import load_predictions
from baykus.health.gate import FLAG_ORDER
from baykus.health.suite import HealthError, gate_name, summary_path
from baykus.robustness.corruptions import suite_conditions

HARMFUL = ("noise-20", "misalign-20", "misalign-40")  # Kabul 2'nin Z kümesi (ölçümden önce)
CAMERA_LOSS = ("drop_rgb", "drop_ir")
K1_LIMIT = -0.005  # temiz val'de fark ≥ −0,5 puan
K3_LIMIT = 0.02  # temiz val'de kapıyı tetikleyen yanlış alarm ≤ %2 görüntü
PREDS = "predictions_coco.json.gz"


def acceptance(clean_diff: float, harmful_ci: list[float | None], clean_false_alarm: float) -> dict:
    lo = harmful_ci[0] if harmful_ci else None
    return {
        "k1": {"value": clean_diff, "limit": K1_LIMIT, "passed": bool(clean_diff >= K1_LIMIT)},
        "k2": {"ci95": list(harmful_ci), "passed": bool(lo is not None and lo > 0)},
        "k3": {
            "value": clean_false_alarm,
            "limit": K3_LIMIT,
            "passed": bool(clean_false_alarm <= K3_LIMIT),
        },
    }


def detection_rates(summaries: dict[str, dict]) -> dict:
    out = {}
    for cid, s in summaries.items():
        n = s["n_images"]

        def rate(k: int, n: int = n) -> float | None:
            return k / n if n else None  # boş koşulda oran hesaplanamaz: null

        out[cid] = {
            "gate_rate": rate(s["gate"]["ir_off"] + s["gate"]["rgb_off"]),
            "rgb": {f: rate(s["rgb"][f]) for f in FLAG_ORDER},
            "ir": {f: rate(s["ir"][f]) for f in FLAG_ORDER},
            "misaligned": rate(s["misaligned"]),
        }
    return out


def check_thresholds(sha: str, records: dict[str, dict | None]) -> None:
    """Sonuçlar bu eşik dosyasıyla mı üretildi? Değilse iki ayar aynı rapora karışmasın."""
    bad = [k for k, r in records.items() if r is not None and r.get("thresholds_sha256") != sha]
    if bad:
        raise HealthError(f"eşik dosyası uyuşmuyor (sha256 {sha[:12]}…): {bad}")


def _load_summaries(results: Path, run_id: str, split: str) -> dict[str, dict]:
    out = {}
    for c in suite_conditions():
        p = summary_path(results, run_id, split, c.id)
        if p.exists():
            out[c.id] = json.loads(p.read_text(encoding="utf-8"))
    if not out:
        raise HealthError(f"{Path(results) / 'health'} altında {run_id} {split} özeti yok")
    shas = {s["thresholds_sha256"] for s in out.values()}
    if len(shas) > 1:
        raise HealthError(f"özetler farklı eşik dosyalarıyla yazılmış: {sorted(shas)}")
    return out


def gate_report(
    *,
    run_id: str,
    split: str,
    robust_preds: Path,
    gt: COCO,
    results: Path,
    n_boot: int = 1000,
    seed: int = 42,
    workers: int = 1,
    thresholds: Path | None = None,
) -> dict:
    summaries = _load_summaries(Path(results), run_id, split)
    if thresholds is not None:
        check_thresholds(hashlib.sha256(Path(thresholds).read_bytes()).hexdigest(), summaries)
    files: dict[str, tuple[Path, Path | None]] = {}
    for cid, s in summaries.items():
        base = Path(robust_preds) / run_name(run_id, split, condition={"id": cid}) / PREDS
        gate = (
            Path(robust_preds) / gate_name(run_id, split, cid) / PREDS if s["evaluated"] else None
        )
        files[cid] = (base, gate)
    missing = [
        str(p) for b, g in files.values() for p in (b, g) if p is not None and not p.exists()
    ]
    if missing:  # uzun bootstrap'tan önce dur
        raise HealthError("eksik tahmin dosyası: " + ", ".join(missing))

    detection = detection_rates(summaries)
    rows, z_gate, z_base = [], [], []
    for cid, (base, gate) in files.items():
        mb = collect_matches(gt, load_predictions(base))
        mg = collect_matches(gt, load_predictions(gate)) if gate else mb
        if gate:
            d = paired_diff_ci(mg, mb, n_boot, seed, workers)
        else:  # aynı girdi, aynı tahminler: fark tanım gereği 0
            d = {
                "diff": 0.0,
                "ci95": [0.0, 0.0] if n_boot else [None, None],
                "significant": False if n_boot else None,
            }
        s = summaries[cid]
        rows.append(
            {
                "condition": cid,
                "n_images": s["n_images"],
                "gate_rate": detection[cid]["gate_rate"],
                "changed": s["changed"],
                "base_map50": ap50(mb),
                "gate_map50": ap50(mg),
                "diff": d["diff"],
                "ci95": d["ci95"],
                "significant": d["significant"],
            }
        )
        if cid in HARMFUL:
            z_gate.append(mg)
            z_base.append(mb)
        print(
            f"[{cid}] E3c={rows[-1]['base_map50']:.4f} kapı={rows[-1]['gate_map50']:.4f}",
            flush=True,
        )

    z = [c for c in HARMFUL if c in summaries]
    harmful = (
        {"conditions": z, **paired_mean_diff_ci(z_gate, z_base, n_boot, seed, workers)}
        if z
        else None
    )
    acc = None
    if split == "val" and "clean" in summaries and z == list(HARMFUL):
        clean = next(r for r in rows if r["condition"] == "clean")
        acc = acceptance(clean["diff"], harmful["ci95"], detection["clean"]["gate_rate"])
    return {
        "run_id": run_id,
        "split": split,
        "label_set": "ir",
        "n_boot": n_boot,
        "seed": seed,
        "thresholds_sha256": next(iter(summaries.values()))["thresholds_sha256"],
        "rows": rows,
        "harmful": harmful,
        "detection": detection,
        "acceptance": acc,
    }
