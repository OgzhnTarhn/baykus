"""Aydınlatma etiketi: RGB parlaklığından, resmi dağılıma uydurulmuş eşiklerle (FR-D4).

Varsayım (ADR 0005): DroneVehicle'ın Gündüz/Gece/Karanlık gece ayrımı RGB parlaklığıyla
monoton ilişkili. Doğrulama: her seviyeden örnek mozaik + kullanıcının gözle kontrolü.
"""

from __future__ import annotations

import cv2
import numpy as np

LEVELS = ("dark_night", "night", "day")
OFFICIAL_COUNTS = {"day": 14478, "night": 5468, "dark_night": 8493}


def mean_brightness(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 2].mean())


def fit_thresholds(values: np.ndarray, counts: dict[str, int]) -> tuple[float, float]:
    v = np.sort(np.asarray(values, dtype=float))
    total = sum(counts.values())
    n_dark = round(len(v) * counts["dark_night"] / total)
    n_night = round(len(v) * counts["night"] / total)

    def cut(k: int) -> float:
        if k <= 0:
            return float(v[0])
        if k >= len(v):
            return float(v[-1]) + 1.0
        return float((v[k - 1] + v[k]) / 2)

    return cut(n_dark), cut(n_dark + n_night)


def assign(values: np.ndarray, thresholds: tuple[float, float]) -> list[str]:
    t_dark, t_night = thresholds
    return ["dark_night" if x < t_dark else "night" if x < t_night else "day" for x in values]
