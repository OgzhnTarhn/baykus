"""Kapı eşiklerinin train'de seçimi (Faz 2 tasarımı §4.1, plan F2 Spec yorumları 8).

Train'den seçilmiş görüntülere sağlamlık paketinin bozulmaları uygulanır (kanca K2), her
durum için "zararlı şiddetler" ile "temiz + zararsız şiddetler"i en iyi ayıran eşik Youden J
ile seçilir. Pozitif / negatif kümeler ölçümden önce sabit (STATES). Val ve test dokunulmaz.
"""

from __future__ import annotations

import math
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from baykus.health.features import frame_features
from baykus.health.gate import Thresholds, assess
from baykus.io import imread_any
from baykus.robustness.corruptions import apply_condition, suite_conditions


@dataclass(frozen=True)
class StateSpec:
    feature: str
    direction: Literal["above", "below"]  # above: x ≥ eşik pozitif; below: x ≤ eşik pozitif
    positives: tuple[str, ...]
    negatives: tuple[str, ...] | None  # None: pozitifler dışındaki bütün koşullar


# Zarar düzeyleri Faz 1 val ölçümünden (tasarım §4.1): σ 20 ve ≥ 20 px zararlı, σ 10 ve 10 px değil
STATES: dict[str, StateSpec] = {
    "rgb_lost": StateSpec("rgb_std", "below", ("drop_rgb",), None),
    "ir_lost": StateSpec("ir_std", "below", ("drop_ir",), None),
    "ir_noisy": StateSpec("ir_noise", "above", ("noise-20",), ("clean", "noise-5", "noise-10")),
    "misaligned": StateSpec(
        "misalign_px",
        "above",
        ("misalign-20", "misalign-40"),
        ("clean", "misalign-5", "misalign-10"),
    ),
    # bilgi durumları: kapıya girmez, oranları raporlanır
    "rgb_blurry": StateSpec("rgb_blur", "below", ("blur-7", "blur-11"), ("clean", "blur-3")),
    "ir_blurry": StateSpec("ir_blur", "below", ("blur-7", "blur-11"), ("clean", "blur-3")),
}


def _clean(values: Sequence[float | None]) -> np.ndarray:
    return np.array([v for v in values if v is not None and math.isfinite(v)], np.float64)


def youden(pos: Sequence[float | None], neg: Sequence[float | None], direction: str) -> dict:
    """J = TPR − FPR'yi en büyük yapan eşik; adaylar sıralı tekil değerlerin orta noktaları.
    Eşitlikte daha düşük FPR, o da eşitse alarm yönünde daha uzak eşik (daha az alarm)."""
    if direction not in ("above", "below"):
        raise ValueError(f"yön 'above' ya da 'below' olmalı: {direction!r}")
    p, n = _clean(pos), _clean(neg)
    if not len(p) or not len(n):
        raise ValueError(f"pozitif ({len(p)}) ve negatif ({len(n)}) örnek gerekli")
    u = np.unique(np.concatenate([p, n]))
    if len(u) < 2:
        raise ValueError("bütün değerler aynı; eşik seçilemez")
    cands = (u[:-1] + u[1:]) / 2
    if direction == "above":
        tpr = (p[None, :] >= cands[:, None]).mean(axis=1)
        fpr = (n[None, :] >= cands[:, None]).mean(axis=1)
        far = cands  # büyük eşik = daha az alarm
    else:
        tpr = (p[None, :] <= cands[:, None]).mean(axis=1)
        fpr = (n[None, :] <= cands[:, None]).mean(axis=1)
        far = -cands
    j = tpr - fpr
    best = max(range(len(cands)), key=lambda i: (j[i], -fpr[i], far[i]))
    return {
        "threshold": float(cands[best]),
        "tpr": float(tpr[best]),
        "fpr": float(fpr[best]),
        "j": float(j[best]),
        "n_pos": int(len(p)),
        "n_neg": int(len(n)),
    }


def _read(path: Path, flag: int) -> np.ndarray:
    img = imread_any(path, flag)
    if img is None:
        raise OSError(f"okunamadı: {path}")
    return img


def _stem_rows(job: tuple) -> list[dict]:
    root, split, stem, seed = job
    bgr = _read(root / "rgb/images" / split / f"{stem}.jpg", cv2.IMREAD_COLOR)
    ir = _read(root / "ir/images" / split / f"{stem}.png", cv2.IMREAD_GRAYSCALE)
    rows = []
    for cond in suite_conditions():
        b, i = apply_condition(bgr, ir, cond, seed=seed, stem=stem)
        rows.append({"stem": stem, "condition": cond.id, **frame_features(b, i)})
    return rows


def feature_rows(
    root: Path, split: str, stems: Sequence[str], seed: int, workers: int = 1
) -> list[dict]:
    """Her stem × paket koşulu için özellik satırı; sıra (stem, SUITE sırası)."""
    jobs = [(Path(root), split, s, seed) for s in stems]
    if workers <= 1:
        parts = [_stem_rows(j) for j in jobs]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            parts = list(ex.map(_stem_rows, jobs, chunksize=8))
    return [r for part in parts for r in part]


def fit_thresholds(rows: list[dict], dark_brightness: float) -> dict:
    states = {}
    for name, s in STATES.items():
        pos = [r[s.feature] for r in rows if r["condition"] in s.positives]
        if s.negatives is None:
            neg = [r[s.feature] for r in rows if r["condition"] not in s.positives]
        else:
            neg = [r[s.feature] for r in rows if r["condition"] in s.negatives]
        res = youden(pos, neg, s.direction)
        states[name] = {
            "feature": s.feature,
            "direction": s.direction,
            "positives": list(s.positives),
            "negatives": list(s.negatives) if s.negatives is not None else None,
            "threshold": res["threshold"],
            "train": res,
        }
    th = Thresholds(
        **{k: v["threshold"] for k, v in states.items()}, dark=dark_brightness, sha256=""
    )
    clean = [r for r in rows if r["condition"] == "clean"]
    fired = sum(assess(r, th)["gate"] != "none" for r in clean)
    return {
        "states": states,
        "dark_brightness": dark_brightness,
        "train_clean_gate_rate": fired / len(clean) if clean else None,
    }
