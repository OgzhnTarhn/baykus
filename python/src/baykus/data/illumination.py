"""Aydınlatma etiketi: RGB parlaklığından (FR-D4, ADR 0005).

- Karanlık gece / gece sınırı: resmi karanlık gece oranına uydurulur (mozaikle doğrulandı).
- Gece / gündüz sınırı: sabit DAY_MIN_BRIGHTNESS. Resmi orana uydurulan eşik (~82,6) aydınlık
  gece karelerini gündüz sayıyordu; 160 elle etiketli örnekte %77,5 → %95,0 (ADR 0005).
"""

from __future__ import annotations

import cv2
import numpy as np

LEVELS = ("dark_night", "night", "day")
OFFICIAL_COUNTS = {"day": 14478, "night": 5468, "dark_night": 8493}
DAY_MIN_BRIGHTNESS = 114.14  # ADR 0005: 160 elle etiketli örnekte en iyi tek eşik
METHOD = "t_dark: resmi karanlık gece oranı; t_day: sabit, 160 etiketli örnek (ADR 0005)"


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


def thresholds(values: np.ndarray) -> tuple[float, float]:
    """(t_dark, t_day). t_day hiçbir zaman t_dark'ın altına inmez."""
    t_dark = fit_thresholds(values, OFFICIAL_COUNTS)[0]
    return t_dark, max(DAY_MIN_BRIGHTNESS, t_dark)
