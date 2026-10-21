"""Hareketli / duran ölçümü (Faz 4 tasarımı §6.3; paydalar ölçümden önce sabit).

İki satır: **oracle** (GT kutuları + GT kimlikleri takip yerine, ego maskesi GT; ürünün ego, nesil,
yeniden konumlama ve hareket modülü) ve **motor** (C++ dizi koşucusunun `tracks[]`'ı; her karede
kimlikli izler GT kutularıyla IoU ≥ 0,5 Macar eşleşmesi).

Etiketli nüfus = referans etiketi `moving` ya da `stationary` olan GT örnekleri; her biri için ürün
sonucu `moving` / `stationary` / `unknown` / `missed` (eşleşen kimlikli iz yok). `unknown` ve
`missed` paydada. Bootstrap iz düzeyinde (1000, seed 42): yalnız dizi içi örnekleme belirsizliği,
genelleme güvencesi değil.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from baykus.motion.audit import DECIDED, Sample, wilson
from baykus.motion.config import MotionCfg
from baykus.motion.ego import Ego
from baykus.motion.images import Box, half_gray
from baykus.motion.labels import Frames, LabelRow
from baykus.motion.reloc import Relocalizer
from baykus.motion.state import MotionState, TrackObs

K = 15
IOU_MIN = 0.5
K9_BA, K9_R = 0.85, 0.80  # Kabul 9 (oracle işlev kapısı, tasarım §6.3)
BOOT_N, BOOT_SEED = 1000, 42
PREDS = ("moving", "stationary", "unknown", "missed")

Key = tuple[str, int, int]  # (dizi, kare, GT iz)


class FrameOut(NamedTuple):
    frame: int
    ego: dict
    tracks: list[dict]  # {"track_id", "box": (x1, y1, x2, y2), "motion", "map_xy": (x, y) | None}


# --- koşular ----------------------------------------------------------------------------------


def _boxes_at(gt: dict[int, dict[int, Box]]) -> dict[int, dict[int, Box]]:
    out: dict[int, dict[int, Box]] = defaultdict(dict)
    for tid, per in gt.items():
        for f, b in per.items():
            out[f][tid] = b
    return out


def oracle_run(frames: Frames, gt: dict[int, dict[int, Box]], cfg: MotionCfg) -> list[FrameOut]:
    """GT kutuları ve kimlikleriyle ürünün ego + hareket modülü (IR, her karede `frame`)."""
    ego, ms = Ego(cfg, Relocalizer(cfg)), MotionState(cfg)
    at = _boxes_at(gt)
    out = []
    for f in frames.frames:
        boxes = at.get(f, {})
        e = ego.update(half_gray(frames.ir(f)), "ir", list(boxes.values()), f)
        motions = ms.update(e, [TrackObs(t, b) for t, b in boxes.items()])
        tracks = [
            {
                "track_id": t,
                "box": b,
                "motion": motions[t].motion,
                "speed_rel": motions[t].speed_rel,
                "map_xy": motions[t].map_xy,
            }
            for t, b in boxes.items()
        ]
        out.append(FrameOut(f, e.to_json(), tracks))
    return out


def _box_tuple(b) -> Box:
    if isinstance(b, dict):
        return (b["x1"], b["y1"], b["x2"], b["y2"])
    return tuple(b)  # type: ignore[return-value]


def load_engine_run(path: Path) -> tuple[str, list[FrameOut]]:
    """Motor dizi koşucusu çıktısı (`.json` ya da `.json.gz`): `{"seq", "frames": [{"frame", "ego",
    "tracks": [{"track_id", "box": {x1, y1, x2, y2}, "motion", "map_xy"}]}]}` (Görev 13)."""
    raw = Path(path).read_bytes()
    rec = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    frames = []
    for fr in rec["frames"]:
        tracks = [
            {
                **t,
                "box": _box_tuple(t["box"]),
                "map_xy": None if t.get("map_xy") is None else tuple(t["map_xy"]),
            }
            for t in fr["tracks"]
        ]
        frames.append(FrameOut(int(fr["frame"]), fr["ego"], tracks))
    return rec["seq"], frames


# --- eşleştirme -------------------------------------------------------------------------------


def _iou(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match_frame(gt_boxes: dict[int, Box], tracks: list[dict], matched: bool) -> dict[int, int]:
    """GT iz → çıktı satırı indeksi. Oracle: aynı kimlik; motor: IoU ≥ 0,5 Macar."""
    if not matched:
        idx = {t["track_id"]: i for i, t in enumerate(tracks)}
        return {g: idx[g] for g in gt_boxes if g in idx}
    gids = list(gt_boxes)
    if not gids or not tracks:
        return {}
    m = np.array([[_iou(gt_boxes[g], t["box"]) for t in tracks] for g in gids])
    rows, cols = linear_sum_assignment(m, maximize=True)
    return {gids[r]: int(c) for r, c in zip(rows, cols, strict=True) if m[r, c] >= IOU_MIN}


class _Matched(NamedTuple):
    preds: dict[Key, str]  # bütün GT örnekleri
    outputs: list[tuple[str, int, str, Key | None]]  # (dizi, kare, motion, eşleşen GT)
    frames: int


def _match_runs(runs: dict[str, list[FrameOut]], gts: dict, matched: bool) -> _Matched:
    preds: dict[Key, str] = {}
    outputs: list[tuple[str, int, str, Key | None]] = []
    n_frames = 0
    for seq, run in runs.items():
        at = _boxes_at(gts.get(seq, {}))
        by_frame = {fr.frame: fr for fr in run}
        for f, boxes in at.items():
            fr = by_frame.get(f)
            pairs = _match_frame(boxes, fr.tracks, matched) if fr else {}
            for g in boxes:
                preds[(seq, f, g)] = fr.tracks[pairs[g]]["motion"] if g in pairs else "missed"
        for fr in run:
            n_frames += 1
            pairs = _match_frame(at.get(fr.frame, {}), fr.tracks, matched)
            back = {i: g for g, i in pairs.items()}
            for i, t in enumerate(fr.tracks):
                key = (seq, fr.frame, back[i]) if i in back else None
                outputs.append((seq, fr.frame, t["motion"], key))
    return _Matched(preds, outputs, n_frames)


# --- ölçüler ----------------------------------------------------------------------------------


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def _mean2(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else (a + b) / 2


def _block(pairs: list[tuple[str, str]]) -> dict:
    """(GT etiketi, ürün sonucu) çiftleri → geri çağırma, BA, kapsam, karışıklık."""
    conf: dict[str, Counter] = {lab: Counter() for lab in DECIDED}
    for lab, pred in pairs:
        conf[lab][pred] += 1
    n_mov, n_sta = sum(conf["moving"].values()), sum(conf["stationary"].values())
    r_mov = _rate(conf["moving"]["moving"], n_mov)
    r_sta = _rate(conf["stationary"]["stationary"], n_sta)
    dec_mov = conf["moving"]["moving"] + conf["moving"]["stationary"]
    dec_sta = conf["stationary"]["moving"] + conf["stationary"]["stationary"]
    return {
        "n": n_mov + n_sta,
        "n_moving": n_mov,
        "n_stationary": n_sta,
        "R_mov": r_mov,
        "R_sta": r_sta,
        "BA": _mean2(r_mov, r_sta),
        "coverage": _rate(dec_mov + dec_sta, n_mov + n_sta),
        "BA_decided": _mean2(
            _rate(conf["moving"]["moving"], dec_mov),
            _rate(conf["stationary"]["stationary"], dec_sta),
        ),
        "confusion": {lab: {p: c[p] for p in PREDS if c[p]} for lab, c in conf.items()},
    }


def _transitions(labels: list[LabelRow]) -> dict[tuple[str, int], list[tuple[int, str]]]:
    """İz başına (kare, yeni etiket): kararlı etiketin bir öncekinden farklı olduğu ilk kare."""
    by_track: dict[tuple[str, int], list[LabelRow]] = defaultdict(list)
    for r in labels:
        if r.label in DECIDED:
            by_track[(r.seq, r.track_id)].append(r)
    out: dict[tuple[str, int], list[tuple[int, str]]] = {}
    for tr, rows in by_track.items():
        rows.sort(key=lambda r: r.frame)
        out[tr] = [
            (b.frame, b.label) for a, b in zip(rows, rows[1:], strict=False) if a.label != b.label
        ]
    return out


def _pct(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def _delay(trans: dict, preds: dict[Key, str], gts: dict) -> dict:
    reached: list[int] = []
    missed = 0
    for (seq, tid), events in trans.items():
        frames = sorted(gts.get(seq, {}).get(tid, {}))
        for i, (t, lab) in enumerate(events):
            end = events[i + 1][0] if i + 1 < len(events) else math.inf
            hit = next(
                (f for f in frames if t <= f < end and preds.get((seq, f, tid)) == lab), None
            )
            if hit is None:
                missed += 1
            else:
                reached.append(hit - t)
    return {
        "transitions": len(reached) + missed,
        "reached": len(reached),
        "not_reached": missed,
        "median": _pct(reached, 50),
        "p90": _pct(reached, 90),
    }


def _bootstrap(pairs_by_track: dict[tuple[str, int], list[tuple[str, str]]]) -> dict:
    keys = sorted(pairs_by_track)
    cnt = np.zeros((len(keys), 4))  # mov_n, mov_hit, sta_n, sta_hit
    for i, k in enumerate(keys):
        for lab, pred in pairs_by_track[k]:
            j = 0 if lab == "moving" else 2
            cnt[i, j] += 1
            cnt[i, j + 1] += pred == lab
    rng = np.random.default_rng(BOOT_SEED)
    stats: dict[str, list[float]] = {"BA": [], "R_mov": [], "R_sta": []}
    for _ in range(BOOT_N):
        if not keys:
            break
        s = cnt[rng.integers(0, len(keys), len(keys))].sum(axis=0)
        r_mov = s[1] / s[0] if s[0] else None
        r_sta = s[3] / s[2] if s[2] else None
        for name, v in (("R_mov", r_mov), ("R_sta", r_sta), ("BA", _mean2(r_mov, r_sta))):
            if v is not None:
                stats[name].append(float(v))
    ci = {k: [_pct(v, 2.5), _pct(v, 97.5)] for k, v in stats.items()}
    return {
        **ci,
        "n_tracks": len(keys),
        "resamples": BOOT_N,
        "seed": BOOT_SEED,
        "note": "iz düzeyinde, yalnız dizi içi örnekleme belirsizliği; genelleme güvencesi değil",
    }


def _precision(m: _Matched, labels: dict[Key, LabelRow]) -> dict:
    out: dict = {}
    for kind in DECIDED:
        c = Counter()
        for _, _, motion, key in m.outputs:
            if motion != kind:
                continue
            c["total"] += 1
            if key is None:
                c["unmatched"] += 1
                continue
            lab = labels[key].label if key in labels else None
            if lab == kind:
                c["correct"] += 1
            elif lab in DECIDED:
                c["wrong_label"] += 1
            else:
                c["excluded"] += 1
        out[kind] = {
            **{k: c[k] for k in ("total", "correct", "wrong_label", "excluded", "unmatched")},
            "P": _rate(c["correct"], c["total"]),
        }
    wrong = out["moving"]["wrong_label"] + out["moving"]["unmatched"]
    out["false_moving_per_frame"] = _rate(wrong, m.frames)
    return out


def score(
    labels: list[LabelRow],
    runs: dict[str, list[FrameOut]],
    gts: dict[str, dict[int, dict[int, Box]]],
    matched: bool,
    k: int = K,
) -> dict:
    m = _match_runs(runs, gts, matched)
    by_key = {(r.seq, r.frame, r.track_id): r for r in labels}
    trans = _transitions(labels)
    labeled, stable = [], []
    by_track: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    per_seq: dict[str, list[tuple[str, str]]] = defaultdict(list)
    excluded: dict[str, Counter] = defaultdict(Counter)
    for key, r in sorted(by_key.items()):
        pred = m.preds.get(key, "missed")
        if r.label not in DECIDED:
            excluded[r.label][pred] += 1
            continue
        pair = (r.label, pred)
        labeled.append(pair)
        per_seq[r.seq].append(pair)
        by_track[(r.seq, r.track_id)].append(pair)
        if all(abs(r.frame - t) > k for t, _ in trans.get((r.seq, r.track_id), [])):
            stable.append(pair)
    return {
        "labeled": _block(labeled),
        "stable": _block(stable),
        "per_seq": {s: _block(p) for s, p in sorted(per_seq.items())},
        "precision": _precision(m, by_key),
        "excluded": {
            lab: {"n": sum(c.values()), "pred": dict(sorted(c.items()))}
            for lab, c in sorted(excluded.items())
        },
        "delay": _delay(trans, m.preds, gts),
        "bootstrap": _bootstrap(by_track),
        "matched": matched,
        "iou_min": IOU_MIN if matched else None,
        "k": k,
    }


def k9(s: dict) -> dict:
    lab = s["labeled"]
    ba, r_mov, r_sta = lab["BA"], lab["R_mov"], lab["R_sta"]
    ok = None not in (ba, r_mov, r_sta) and ba >= K9_BA and r_mov >= K9_R and r_sta >= K9_R
    return {"BA": ba, "R_mov": r_mov, "R_sta": r_sta, "pass": bool(ok)}


def vs_human(
    samples: dict[str, Sample],
    answers: dict[str, str],
    runs: dict[str, list[FrameOut]],
    gts: dict,
    matched: bool,
) -> dict:
    """Kör örneklemde insanın kararı (emin değilim hariç) ile ürünün kararı; Wilson %95 GA."""
    m = _match_runs(runs, gts, matched)
    n = agree = unknown = missed = dec_n = dec_agree = 0
    for i, s in samples.items():
        ans = answers.get(i)
        if ans not in DECIDED:
            continue
        pred = m.preds.get((s.seq, s.frame, s.track_id), "missed")
        n += 1
        agree += pred == ans
        unknown += pred == "unknown"
        missed += pred == "missed"
        if pred in DECIDED:
            dec_n += 1
            dec_agree += pred == ans
    return {
        "n": n,
        "agree": agree,
        "rate": _rate(agree, n),
        "ci95": list(wilson(agree, n)),
        "product_unknown": unknown,
        "product_missed": missed,
        "decided": {
            "n": dec_n,
            "agree": dec_agree,
            "rate": _rate(dec_agree, dec_n),
            "ci95": list(wilson(dec_agree, dec_n)),
        },
    }


# --- ego ve donmuş kare -----------------------------------------------------------------------


def ego_quality(run: list[FrameOut], relink_max_gap: int) -> dict:
    """`ok` oranı, anahtar kare, nesiller, birleşmeler, kısa kayıptan dönüşte bağlama oranı."""
    ok = [fr.ego["ok"] for fr in run]
    attempts = success = gap = 0
    for fr, o in zip(run, ok, strict=True):
        if not o:
            gap += 1
            continue
        if 0 < gap <= relink_max_gap:
            attempts += 1
            success += bool(fr.ego.get("relinked"))
        gap = 0
    inl = [fr.ego["inliers"] for fr in run if fr.ego["ok"] and fr.ego.get("inliers") is not None]
    return {
        "frames": len(run),
        "ok_rate": _rate(sum(ok), len(run)),
        "keyframes": sum(bool(fr.ego.get("keyframe")) for fr in run),
        "map_gens": sorted({fr.ego["map_gen"] for fr in run if fr.ego["ok"]}),
        "merges": sum(fr.ego.get("merge") is not None for fr in run),
        "relink": {"attempts": attempts, "success": success, "rate": _rate(success, attempts)},
        "inliers_median": _pct(inl, 50),
    }


def frozen_rates(seq, cfg: MotionCfg) -> dict:
    """Temiz dizide kanal başına donmuş kare ve alarm oranı; `freeze@K` (K = ⌊n/2⌋'den sonra aynı
    kare) gecikmesi: K'den sonra alarmın ilk açıldığı kareye kadar kare sayısı."""
    from baykus.health.frozen import FrozenDetector

    halves = [(half_gray(seq.rgb(f)), half_gray(seq.ir(f))) for f in seq.frames]
    fd = FrozenDetector(cfg)
    clean = [fd.update(r, i, False, False) for r, i in halves]
    kk = len(halves) // 2
    fd = FrozenDetector(cfg)
    frozen_seq = halves[: kk + 1] + [halves[kk]] * (len(halves) - kk - 1)
    froz = [fd.update(r, i, False, False) for r, i in frozen_seq]
    out = {"frames": len(halves), "freeze_at": kk}
    for ch in ("rgb", "ir"):
        first = next((j - kk for j in range(kk + 1, len(froz)) if froz[j][ch]["alarm"]), None)
        out[ch] = {
            "frozen_rate": _rate(sum(o[ch]["frozen"] for o in clean), len(clean)),
            "alarm_rate": _rate(sum(o[ch]["alarm"] for o in clean), len(clean)),
            "freeze_alarm_delay": first,
        }
    return out
