"""Güven kalibrasyonu (Faz 2 tasarımı §4.3, §6.3; plan F2 Spec yorumları 9).

Skor → "doğru tespit olasılığı" eşlemesi val'de izotonik regresyonla (PAV, bağımlılıksız)
öğrenilir; testte ECE önce / sonra. Etiket: IoU ≥ 0,5'te eşleşti (collect_matches TP); yok
sayılan GT'ye düşen tespit ne doğru ne yanlış. Tablo düz JSON (x: blok üst sınırları, y:
değerler): Faz 3'te C++ motoru aynı adım fonksiyonunu uygular.
"""

from __future__ import annotations

import numpy as np
from pycocotools.coco import COCO

from baykus.eval.bootstrap import collect_matches
from baykus.eval.coco_eval import evaluate_coco

ECE_BINS = 15
CONF_MIN = 0.25  # operatör eşiği (§15.3); ECE bu ham skor üstündeki tespitlerde
K4_LIMIT = 0.001  # |ΔmAP50| ≤ 0,1 puan


def match_labels(gt: COCO, dets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    m = collect_matches(gt, dets)
    scores, labels = [], []
    for cm in m.classes.values():
        keep = cm.tp | cm.fp
        scores.append(cm.scores[keep])
        labels.append(cm.tp[keep].astype(np.int64))
    if not scores:
        return np.zeros(0), np.zeros(0, np.int64)
    return np.concatenate(scores), np.concatenate(labels)


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Havuzlu bitişik ihlaller (PAV). Eşit skorlar tek nokta; dönüş (blok üst sınırları,
    azalmayan değerler), eşit değerli komşu bloklar birleşik."""
    s = np.asarray(scores, np.float64)
    lab = np.asarray(labels, np.float64)
    if not len(s):
        raise ValueError("kalibrasyon için tespit yok")
    xs, inv = np.unique(s, return_inverse=True)
    w = np.bincount(inv).astype(np.float64)
    ys = np.bincount(inv, weights=lab) / w
    vals: list[float] = []
    wts: list[float] = []
    his: list[float] = []
    for x, y, wi in zip(xs, ys, w, strict=True):
        vals.append(float(y))
        wts.append(float(wi))
        his.append(float(x))
        while len(vals) > 1 and vals[-2] >= vals[-1]:  # ihlal ya da eşit: birleştir
            v2, w2, h2 = vals.pop(), wts.pop(), his.pop()
            v1, w1 = vals.pop(), wts.pop()
            his.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            his.append(h2)
    return np.array(his), np.array(vals)


def apply_isotonic(scores: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Blok i, (x[i−1], x[i]] aralığı; en küçüğün altı ilk blok, en büyüğün üstü son blok."""
    idx = np.clip(np.searchsorted(x, np.asarray(scores, np.float64), side="left"), 0, len(x) - 1)
    return y[idx]


def _bin_index(conf: np.ndarray, bins: int) -> np.ndarray:
    return np.minimum((np.clip(conf, 0.0, 1.0) * bins).astype(np.int64), bins - 1)


def reliability(conf: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS) -> list[dict]:
    conf, labels = np.asarray(conf, np.float64), np.asarray(labels, np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = _bin_index(conf, bins)
    out = []
    for b in range(bins):
        sel = idx == b
        n = int(sel.sum())
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": n,
                "conf": float(conf[sel].mean()) if n else None,
                "acc": float(labels[sel].mean()) if n else None,
            }
        )
    return out


def ece(conf: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS) -> float:
    total = len(conf)
    if not total:
        return 0.0
    return float(
        sum(
            b["n"] / total * abs(b["acc"] - b["conf"])
            for b in reliability(conf, labels, bins)
            if b["n"]
        )
    )


def calibrate(
    val_gt: COCO,
    val_dets: list[dict],
    test_gt: COCO,
    test_dets: list[dict],
    conf_min: float = CONF_MIN,
) -> dict:
    vs, vl = match_labels(val_gt, val_dets)
    x, y = fit_isotonic(vs, vl)
    ts, tl = match_labels(test_gt, test_dets)
    keep = ts >= conf_min
    raw, lab = ts[keep], tl[keep]
    cal = apply_isotonic(raw, x, y)
    new_scores = apply_isotonic(np.array([d["score"] for d in test_dets]), x, y)
    test_cal = [{**d, "score": float(s)} for d, s in zip(test_dets, new_scores, strict=True)]
    before = evaluate_coco(test_gt, test_dets)["map50"]
    after = evaluate_coco(test_gt, test_cal)["map50"]
    delta = after - before
    return {
        "table": {"x": x.tolist(), "y": y.tolist()},
        "conf_min": conf_min,
        "bins": ECE_BINS,
        "val": {"n_dets": int(len(vs))},
        "test": {
            "n_eval": int(keep.sum()),
            "ece_before": ece(raw, lab),
            "ece_after": ece(cal, lab),
            "map50_before": before,
            "map50_after": after,
            "delta_map50": delta,
            "bins_before": reliability(raw, lab),
            "bins_after": reliability(cal, lab),
        },
        "k4": {"value": abs(delta), "limit": K4_LIMIT, "passed": bool(abs(delta) <= K4_LIMIT)},
    }
