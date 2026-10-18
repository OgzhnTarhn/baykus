"""Hizasızlık alarmının kare kuralı: `r_min` seçimi (train) ve val kare birleştirme (Faz 3
tasarımı §4.2, plan F3 Görev 6).

`r_min` yalnız alarm kaldırabilir: `misaligned_frame` = Faz 2 kuralı ve `response ≥ r_min`.
Seçim train özellik satırlarından (`data/health/train_features.csv`); val'de yeni koşu yok, sonuç
Faz 2'nin kapılı ve taban tahminlerinden kare kare birleştirilir ("val'de keşif, PyTorch E3c").
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pycocotools.coco import COCO

from baykus.eval.bootstrap import ap50, collect_matches, paired_diff_ci, paired_mean_diff_ci
from baykus.eval.cli import run_name
from baykus.eval.predictions import load_predictions
from baykus.health.compare import HARMFUL, PREDS, _load_summaries, acceptance
from baykus.health.features import FEATURES
from baykus.health.gate import REASONS, Thresholds, assess
from baykus.health.suite import HealthError, gate_name
from baykus.robustness.corruptions import suite_conditions
from baykus.runinfo import git_info, iso, utc_now, write_json

CRITERION = "clean_alarm*50<=n"  # train temiz alarm ≤ %2, tam sayıyla (ölçümden önce sabit)
ALARM_PER = 50
POSITIVE = ("misalign-20", "misalign-40")
TEMPORAL = {"window": 5, "min_frames": 3, "tol_px": 3.0}  # tasarım §4.2, ayarlanmaz
MISALIGNED = REASONS["misaligned"]
HEALTH_LOG = "health.jsonl.gz"
LABEL = "val keşfi, PyTorch E3c"


def _features(row: dict) -> dict[str, float | None]:
    """CSV satırı (metin) ya da sözlük → özellikler; boş hücre None."""
    out = {}
    for k in FEATURES:
        v = row.get(k)
        out[k] = None if v is None or v == "" else float(v)
    return out


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def select_r_min(rows: list[dict], th: Thresholds) -> dict:
    """Ölçütü sağlayan en küçük aday `r` (adaylar 0 + train'deki farklı yanıtlar, artan)."""
    feats = [(r["condition"], _features(r)) for r in rows]
    clean = [f for c, f in feats if c == "clean"]
    if not clean:
        raise ValueError("train özelliklerinde 'clean' satırı yok")
    # Kare başına: r_min'den bağımsız alarm (kayıp, gürültü) ya da yalnız hizasızlık alarmı.
    # Hizasızlık alarmı yanıt ≥ r iken sürer; sayım assess(f, th, r) ile aynıdır.
    fixed, mis_resp = 0, []
    for f in clean:
        if assess(f, th, r_min=float("inf"))["gate"] != "none":
            fixed += 1
        elif assess(f, th)["pair"]["misaligned_frame"]:
            mis_resp.append(f["misalign_response"])
    cands = sorted({0.0} | {f["misalign_response"] for _, f in feats} - {None})
    n = len(clean)
    for r in cands:
        alarms = fixed + sum(v >= r for v in mis_resp)
        if alarms * ALARM_PER <= n:
            break
    else:
        raise ValueError(
            f"r_min bulunamadı: hizasızlık dışı alarmlar ({fixed}/{n}) ölçütü ({CRITERION}) aşıyor"
        )
    pos = [f for c, f in feats if c in POSITIVE]

    def tpr(group: list[dict]) -> float | None:
        return _rate(sum(assess(f, th, r)["pair"]["misaligned"] for f in group), len(group))

    def dark(f: dict) -> bool:
        return f["rgb_brightness"] is not None and f["rgb_brightness"] < th.dark

    return {
        "r_min": float(r),
        "criterion": CRITERION,
        "train_clean_alarm_rate": alarms / n,
        "train_tpr": {
            "all": tpr(pos),
            "dark": tpr([f for f in pos if dark(f)]),
            "bright": tpr([f for f in pos if not dark(f)]),
        },
        "n_clean": n,
        "n_pos": len(pos),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_gate_model(out: Path, sel: dict, th: Thresholds, features_csv: Path) -> Path:
    return write_json(
        Path(out),
        {
            **sel,
            "temporal": dict(TEMPORAL),
            "thresholds_sha256": th.sha256,
            "train_features_sha256": _sha256(features_csv),
            "git_commit": git_info()["git_commit"],
            "created_at": iso(utc_now()),
        },
    )


@dataclass(frozen=True)
class GateModel:
    r_min: float
    window: int
    min_frames: int
    tol_px: float
    sha256: str
    thresholds_sha256: str


def load_gate_model(path: Path) -> GateModel:
    """gate_model.json → GateModel; sha256 dosya baytlarından."""
    data = Path(path).read_bytes()
    raw = json.loads(data.decode("utf-8"))
    t = raw.get("temporal") or {}
    missing = [k for k in ("r_min", "temporal", "thresholds_sha256") if k not in raw]
    missing += [f"temporal.{k}" for k in TEMPORAL if k not in t]
    if missing:
        raise ValueError(f"{path}: kapı modeli dosyasında eksik: {missing}")
    return GateModel(
        r_min=float(raw["r_min"]),
        window=int(t["window"]),
        min_frames=int(t["min_frames"]),
        tol_px=float(t["tol_px"]),
        sha256=hashlib.sha256(data).hexdigest(),
        thresholds_sha256=str(raw["thresholds_sha256"]),
    )


def _by_image(preds: list[dict]) -> dict[int, list[tuple]]:
    out: dict[int, list[tuple]] = {}
    for p in preds:
        key = (p["category_id"], tuple(p["bbox"]), p["score"])
        out.setdefault(p["image_id"], []).append(key)
    return {k: sorted(v) for k, v in out.items()}


def _removed(h: dict, r_min: float) -> bool:
    resp = h["pair"].get("response")
    return h.get("reason") == MISALIGNED and (resp is None or resp < r_min)


def splice_condition(
    health_rows: list[dict],
    gate_preds: list[dict] | None,
    base_preds: list[dict],
    stem_to_id: dict[str, int],
    r_min: float,
    evaluated: bool,
) -> tuple[list[dict], dict]:
    """Kare kuralıyla kalkan alarmların karesi taban, diğerleri kapılı tahminden.

    Ön koşul: kapının dokunmadığı (`changed` False) her karede kapılı ve taban tahminleri dosyadaki
    değerleriyle eşit; değilse birleştirme yapılmaz (`precondition_ok` False, tahmin listesi boş).
    """
    unknown = [h["stem"] for h in health_rows if h["stem"] not in stem_to_id]
    if unknown:
        raise ValueError(f"COCO'da olmayan kareler: {unknown[:5]}")
    if not evaluated:  # Faz 2 kapısı hiçbir pikseli değiştirmedi: kapılı = taban
        return list(base_preds), {"removed": 0, "precondition_ok": True, "source": "base"}
    if gate_preds is None:
        raise ValueError("değerlendirilmiş koşulda kapılı tahminler gerekli")
    gb, bb = _by_image(gate_preds), _by_image(base_preds)
    untouched = [stem_to_id[h["stem"]] for h in health_rows if not h["changed"]]
    if any(gb.get(i, []) != bb.get(i, []) for i in untouched):
        return [], {"removed": None, "precondition_ok": False, "source": None}
    removed = {stem_to_id[h["stem"]] for h in health_rows if _removed(h, r_min)}
    preds = [p for p in base_preds if p["image_id"] in removed]
    preds += [p for p in gate_preds if p["image_id"] not in removed]
    return preds, {"removed": len(removed), "precondition_ok": True, "source": "spliced"}


def download_list(results: Path, run_id: str) -> list[str]:
    """Drive'daki sağlamlık klasöründen (robust/) laptopa inecek yollar, val koşulları."""
    out = []
    for c in suite_conditions():
        sp = Path(results) / "health" / f"{run_id}_val_{c.id}_health.json"
        if not sp.exists():
            continue
        evaluated = json.loads(sp.read_text(encoding="utf-8"))["evaluated"]
        base = run_name(run_id, "val", condition={"id": c.id})
        gate = gate_name(run_id, "val", c.id)
        out += [f"{base}/{PREDS}", f"{gate}/{HEALTH_LOG}"]
        if evaluated:
            out.append(f"{gate}/{PREDS}")
    return out


def _read_health(path: Path) -> list[dict]:
    text = gzip.decompress(Path(path).read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _catch(frames: list[dict], r_min: float, dark: float) -> dict:
    """Kare kuralından sonra hizasızlık yakalama oranı (karanlık / aydınlık)."""

    def rate(fs: list[dict]) -> float | None:
        return _rate(sum(f["pair"]["misaligned"] and not _removed(f, r_min) for f in fs), len(fs))

    def is_dark(f: dict) -> bool:
        b = f["rgb"].get("brightness")
        return b is not None and b < dark

    return {
        "all": rate(frames),
        "dark": rate([f for f in frames if is_dark(f)]),
        "bright": rate([f for f in frames if not is_dark(f)]),
    }


def guard_val_report(
    *,
    run_id: str,
    robust_preds: Path,
    gt: COCO,
    results: Path,
    gate_model: GateModel,
    th: Thresholds,
    n_boot: int = 1000,
    seed: int = 42,
    workers: int = 1,
) -> dict:
    """Faz 2 `gate_report` satırları, kare kuralıyla birleştirilmiş kapılı tahminlerle."""
    split = "val"
    summaries = _load_summaries(Path(results), run_id, split)
    sha = next(iter(summaries.values()))["thresholds_sha256"]
    if sha != th.sha256 or gate_model.thresholds_sha256 != th.sha256:
        raise HealthError("eşik dosyası, Faz 2 özetleri ve kapı modeli aynı eşiklerden değil")
    # file_name = stem (baykus.data.coco)
    stem_to_id = {img["file_name"]: img["id"] for img in gt.loadImgs(sorted(gt.getImgIds()))}
    r_min = gate_model.r_min
    rows, z_gate, z_base, catch = [], [], [], {}
    for cid, s in summaries.items():
        base_dir = Path(robust_preds) / run_name(run_id, split, condition={"id": cid})
        gate_dir = Path(robust_preds) / gate_name(run_id, split, cid)
        health = _read_health(gate_dir / HEALTH_LOG)
        base = load_predictions(base_dir / PREDS)
        gate = load_predictions(gate_dir / PREDS) if s["evaluated"] else None
        preds, info = splice_condition(health, gate, base, stem_to_id, r_min, s["evaluated"])
        row = {"condition": cid, "n_images": s["n_images"], **info}
        if cid in POSITIVE:
            catch[cid] = _catch(health, r_min, th.dark)
        if not info["precondition_ok"]:  # "birleştirilemedi": sayı verilmez
            rows.append(row)
            print(f"[{cid}] birleştirilemedi (ön koşul tutmadı)", flush=True)
            continue
        alarms = s["gate"]["ir_off"] + s["gate"]["rgb_off"] - info["removed"]
        mb = collect_matches(gt, base)
        mg = collect_matches(gt, preds) if info["source"] == "spliced" else mb
        if info["source"] == "spliced":
            d = paired_diff_ci(mg, mb, n_boot, seed, workers)
        else:
            d = {
                "diff": 0.0,
                "ci95": [0.0, 0.0] if n_boot else [None, None],
                "significant": False if n_boot else None,
            }
        row.update(
            gate_rate=_rate(alarms, s["n_images"]),
            changed=s["changed"] - info["removed"],
            base_map50=ap50(mb),
            gate_map50=ap50(mg),
            diff=d["diff"],
            ci95=d["ci95"],
            significant=d["significant"],
        )
        rows.append(row)
        if cid in HARMFUL:
            z_gate.append(mg)
            z_base.append(mb)
        print(f"[{cid}] E3c={row['base_map50']:.4f} kapı={row['gate_map50']:.4f}", flush=True)

    ok = [r["condition"] for r in rows if r["precondition_ok"]]
    harmful = None
    if all(c in ok for c in HARMFUL):  # Faz 2 Kabul 2: Z kümesinin hepsi birleştirilebildiyse
        z = [c for c in ok if c in HARMFUL]  # z_gate / z_base ile aynı sıra
        harmful = {"conditions": z, **paired_mean_diff_ci(z_gate, z_base, n_boot, seed, workers)}
    acc = None
    if harmful and "clean" in ok:
        clean = next(r for r in rows if r["condition"] == "clean")
        acc = acceptance(clean["diff"], harmful["ci95"], clean["gate_rate"])
    return {
        "label": LABEL,
        "run_id": run_id,
        "split": split,
        "label_set": "ir",
        "r_min": r_min,
        "gate_model_sha256": gate_model.sha256,
        "thresholds_sha256": th.sha256,
        "n_boot": n_boot,
        "seed": seed,
        "rows": rows,
        "harmful": harmful,
        "misaligned_catch": catch,
        "acceptance": acc,
    }
