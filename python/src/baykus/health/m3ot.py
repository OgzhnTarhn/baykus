"""M3OT dizilerinde sağlık kontrolü ve demo dizisi (Faz 3 tasarımı §4.7, plan F3 Görev 12).

Kareler dizi sırasıyla; zaman kuralı penceresi her dizi ve koşul başında boştur. Koşullar Faz 2
fonksiyonlarıyla (gürültü seed 42, kare adıyla). Üç kural yan yana: Faz 2 (tek kare, r_min yok),
kare kuralı (r_min) ve kare + zaman kuralı. AP50 IR etiketleriyle, sınıfsız, değerlendirme
katmanında (ham ≥ 0,001); `gate=auto` ve aynı bozulmuş girdide `gate=off`.

Hız: `gate=auto` tespiti yalnız kapı kararına bağlıdır (none / ir_off / rgb_off); her kare ve
karar için çıkarım bir kez yapılır, kurallar paylaşır.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from baykus.engine_ref.compose import decode_pair
from baykus.engine_ref.measure import ap50_sequence, load_gt
from baykus.engine_ref.models import Models
from baykus.engine_ref.pipeline import Det, Pipeline
from baykus.health.features import frame_features
from baykus.health.gate import apply_gate, assess
from baykus.health.temporal import MisalignWindow
from baykus.robustness.corruptions import apply_condition, select_conditions

CONDITIONS = ("clean", "misalign-20", "misalign-20@K", "noise-20", "drop_ir")
RULES = ("phase2", "frame", "frame+temporal")
ALARM_IDS = ("rgb_lost", "ir_lost", "ir_noisy", "misaligned")
ALARM_WINDOW, ALARM_MIN = 5, 3  # tasarım §4.4: son 5 karenin en az 3'ü (sabit)
SEED = 42
EVAL_FLOOR = 0.001
DEMO_CLEAN_MAX, DEMO_MISALIGN_MIN, DEMO_DELAY_MAX = 0.02, 0.90, 10  # ölçümden önce sabit


class Alarms:
    """§4.4 alarm sözleşmesi: kayıp ve gürültü son 5 karenin 3'ünde; hizasız = karenin
    `pair.misaligned`'ı (zaman kuralı zaten pencereli). Histerezis yok. UI `lib/alarms.ts` aynı."""

    def __init__(self) -> None:
        self._flags: deque[tuple[bool, bool, bool]] = deque(maxlen=ALARM_WINDOW)

    def reset(self) -> None:
        self._flags.clear()

    def update(self, h: dict) -> list[str]:
        self._flags.append(
            ("lost" in h["rgb"]["flags"], "lost" in h["ir"]["flags"], "noisy" in h["ir"]["flags"])
        )
        counts = [sum(f[i] for f in self._flags) for i in range(3)]
        on = [a for a, c in zip(ALARM_IDS[:3], counts, strict=True) if c >= ALARM_MIN]
        if h["pair"]["misaligned"]:
            on.append("misaligned")
        return on


def alarm_states(healths: list[dict]) -> list[list[str]]:
    a = Alarms()
    return [a.update(h) for h in healths]


def corrupt(
    bgr: np.ndarray, ir: np.ndarray, cond: str, frame: int, k: int, stem: str
) -> tuple[np.ndarray, np.ndarray]:
    """`frame` dizideki sıra (0'dan); `misalign-20@K` yalnız `frame ≥ k`'da kaydırır."""
    if cond not in CONDITIONS:
        raise ValueError(f"bilinmeyen koşul: {cond!r}; seçenekler: {CONDITIONS}")
    if cond == "misalign-20@K":
        if frame < k:
            return bgr.copy(), ir.copy()
        cond = "misalign-20"
    (c,) = select_conditions([cond])
    return apply_condition(bgr, ir, c, seed=SEED, stem=stem)


def delay(misaligned: list[bool], k: int) -> tuple[int | None, bool]:
    """(K ve sonrasındaki ilk alarm − K ya da None, K−1'de alarm açık mıydı)."""
    first = next((i for i in range(k, len(misaligned)) if misaligned[i]), None)
    before = k >= 1 and bool(misaligned[k - 1])
    return (None if first is None else first - k), before


def demo_criterion(misaligned_rates: dict[str, float], delay_frames: int | None) -> dict:
    """Kare + zaman kuralı, `gate=advise`: temizde ≤ %2, misalign-20'de ≥ %90, gecikme ≤ 10."""
    out = {
        "clean_le_2pct": misaligned_rates["clean"] <= DEMO_CLEAN_MAX,
        "misalign_ge_90pct": misaligned_rates["misalign-20"] >= DEMO_MISALIGN_MIN,
        "delay_le_10": delay_frames is not None and delay_frames <= DEMO_DELAY_MAX,
    }
    out["passed"] = all(out.values())
    return out


def _coco(dets: list[Det], image_id: int) -> list[dict]:
    return [
        {
            "image_id": image_id,
            "category_id": 1,
            "bbox": [d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]],
            "score": d.score_raw,
        }
        for d in dets
    ]


def _rate(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _cond_health(
    pipe: Pipeline, frames: list[tuple[np.ndarray, np.ndarray]], seq: str, idx: list[int],
    cond: str, k: int, rules: tuple[str, ...], ids: list[int],
) -> dict:  # fmt: skip
    m = pipe.models
    window = MisalignWindow(m.gate.window, m.gate.min_frames, m.gate.tol_px)
    healths: dict[str, list[dict]] = {r: [] for r in rules}
    auto: dict[str, list[dict]] = {r: [] for r in rules}
    off: list[dict] = []
    for i, ((bgr, ir), f, img_id) in enumerate(zip(frames, idx, ids, strict=True)):
        bgr, ir = corrupt(bgr, ir, cond, i, k, f"{seq}_{f:06d}")
        feats = frame_features(bgr, ir, vector=True)
        frame_h = assess(feats, m.thresholds, m.gate.r_min)
        rule_h = {}
        if "phase2" in rules:
            rule_h["phase2"] = assess(feats, m.thresholds)
        if "frame" in rules:
            rule_h["frame"] = frame_h
        if "frame+temporal" in rules:
            rule_h["frame+temporal"] = window.update(frame_h)
        by_gate: dict[str, list[dict]] = {}  # kapı kararı → bu karenin tespitleri
        for gate in ["none", *(rule_h[r]["gate"] for r in rules)]:
            if gate not in by_gate:
                b, i_ = apply_gate(bgr, ir, gate)
                by_gate[gate] = _coco(pipe.detect(b, i_, EVAL_FLOOR)["dets_all"], img_id)
        off += by_gate["none"]
        for r in rules:
            healths[r].append(rule_h[r])
            auto[r] += by_gate[rule_h[r]["gate"]]
    return {"healths": healths, "auto": auto, "off": off}


def run_sequence(
    seq_dir: Path,
    labels_dir: Path,
    models: Models,
    rules: tuple[str, ...] = RULES,
    *,
    backend=None,
    keep_coco: bool = False,
) -> dict:
    seq_dir = Path(seq_dir)
    index = json.loads((seq_dir / "index.json").read_text(encoding="utf-8"))
    seq, idx = index["name"], list(index["frames"])
    n, k = len(idx), len(idx) // 2
    gt = load_gt(labels_dir, index["split"])
    stem_ids = {img["file_name"]: img["id"] for img in gt.loadImgs(gt.getImgIds())}
    ids = [stem_ids[f"{seq}_{f:06d}"] for f in idx]
    frames = [
        decode_pair(
            (seq_dir / "rgb" / f"{f:06d}.jpg").read_bytes(),
            (seq_dir / "ir" / f"{f:06d}.png").read_bytes(),
        )
        for f in idx
    ]
    pipe = Pipeline(models, backend=backend)
    out_rules: dict[str, dict] = {r: {} for r in rules}
    coco: dict[str, dict] = {}
    for cond in CONDITIONS:
        res = _cond_health(pipe, frames, seq, idx, cond, k, rules, ids)
        ap_off = ap50_sequence(res["off"], gt, ids)
        coco[cond] = {"off": res["off"], "auto": res["auto"]}
        for r in rules:
            hs = res["healths"][r]
            alarms = alarm_states(hs)
            ap_auto = ap50_sequence(res["auto"][r], gt, ids)
            row = {
                "state_rates": {
                    "rgb_lost": _rate(["lost" in h["rgb"]["flags"] for h in hs]),
                    "ir_lost": _rate(["lost" in h["ir"]["flags"] for h in hs]),
                    "ir_noisy": _rate(["noisy" in h["ir"]["flags"] for h in hs]),
                    "misaligned_frame": _rate([h["pair"]["misaligned_frame"] for h in hs]),
                    "misaligned": _rate([h["pair"]["misaligned"] for h in hs]),
                },
                "alarm_rates": {a: _rate([a in s for s in alarms]) for a in ALARM_IDS},
                "gate_rate": _rate([h["gate"] != "none" for h in hs]),
                "ap50_auto": ap_auto,
                "ap50_off": ap_off,
                "ap50_diff": ap_auto - ap_off,
            }
            if cond == "misalign-20@K":
                d, before = delay(["misaligned" in s for s in alarms], k)
                row["delay"], row["alarm_before_onset"] = d, before
            out_rules[r][cond] = row
    result = {"seq": seq, "split": index["split"], "n_frames": n, "k": k, "rules": out_rules}
    if "frame+temporal" in rules:
        t = out_rules["frame+temporal"]
        rates = {c: t[c]["alarm_rates"]["misaligned"] for c in ("clean", "misalign-20")}
        result["demo"] = demo_criterion(rates, t["misalign-20@K"]["delay"])
    if keep_coco:
        result["_coco"] = coco
        result["_ids"] = ids
    return result


def combine(results: list[dict], labels_dir: Path, split: str) -> dict:
    """Dizilerin birleşiği: oranlar kare ağırlıklı, AP50 bütün karelerin tahminleriyle."""
    gt = load_gt(labels_dir, split)
    ids = [i for r in results for i in r["_ids"]]
    n = sum(r["n_frames"] for r in results)
    out: dict[str, dict] = {}
    for rule in results[0]["rules"]:
        out[rule] = {}
        for cond in CONDITIONS:
            rows = [r["rules"][rule][cond] for r in results]
            w = [r["n_frames"] / n for r in results]

            def avg(get, rows=rows, w=w):
                vals = [get(x) for x in rows]
                return None if None in vals else sum(a * b for a, b in zip(vals, w, strict=True))

            off = [d for r in results for d in r["_coco"][cond]["off"]]
            auto = [d for r in results for d in r["_coco"][cond]["auto"][rule]]
            ap_off, ap_auto = ap50_sequence(off, gt, ids), ap50_sequence(auto, gt, ids)
            out[rule][cond] = {
                "alarm_rates": {a: avg(lambda x, a=a: x["alarm_rates"][a]) for a in ALARM_IDS},
                "gate_rate": avg(lambda x: x["gate_rate"]),
                "ap50_auto": ap_auto,
                "ap50_off": ap_off,
                "ap50_diff": ap_auto - ap_off,
            }
    return out


def pick_demo(results: list[dict], config: Path) -> dict:
    """Tasarım §4.6: kuralı geçen dizilerin ad sırasıyla ilki; hiçbiri geçmezse ilk dizi."""
    ordered = sorted(results, key=lambda r: r["seq"])
    passing = [r for r in ordered if r["demo"]["passed"]]
    choice = (passing or ordered)[0]["seq"]
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    cfg["demo"] = choice
    text = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
    Path(config).write_text(text, encoding="utf-8", newline="\n")
    return {"demo": choice, "criterion_passed": bool(passing)}
