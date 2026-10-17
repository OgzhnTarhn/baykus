"""Sağlamlık paketi bozulmaları (spec §12.3, FR-V4, kanca K2).

Saf numpy/OpenCV fonksiyonları: bölümden bağımsız ve seed'li, train'de de kullanılabilir.
RGB bellekte BGR uint8 (H, W, 3), IR gri uint8 (H, W). Gürültü görüntü başına seed'lenir
(seed + dosya adı + koşul): paralel işçi ya da sıra sonucu değiştirmez.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

# ad → (hedef modalite, spec §12.3 şiddetleri; özdeş şiddet dahil)
SUITE: dict[str, tuple[str, tuple[float, ...]]] = {
    "darken": ("rgb", (1.0, 0.5, 0.25, 0.1)),  # piksel çarpanı (gamma eğrisi değil)
    "noise": ("ir", (0, 5, 10, 20)),  # σ, 8-bit
    "blur": ("both", (0, 3, 7, 11)),  # Gauss çekirdeği
    "misalign": ("ir", (0, 5, 10, 20, 40)),  # IR yatay kaydırma, piksel
    "drop_rgb": ("rgb", (1,)),  # kamera kaybı
    "drop_ir": ("ir", (1,)),
}
IDENTITY: dict[str, float] = {"darken": 1.0, "noise": 0, "blur": 0, "misalign": 0}
MODE_NEEDS: dict[str, set[str]] = {
    "rgb": {"rgb"},
    "ir": {"ir"},
    "early_fusion": {"rgb", "ir"},
    "late_fusion": {"rgb", "ir"},  # E4/E4b: GPU koşusu yok, E1 + E2 tahminleri birleşir
}


@dataclass(frozen=True)
class Condition:
    corruption: str  # clean | SUITE anahtarı
    severity: float
    target: str  # none | rgb | ir | both

    @property
    def id(self) -> str:
        if self.corruption in ("clean", "drop_rgb", "drop_ir"):
            return self.corruption
        return f"{self.corruption}-{self.severity:g}"


CLEAN = Condition("clean", 0.0, "none")


def darken(img: np.ndarray, factor: float) -> np.ndarray:
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"darken: çarpan 0-1 arası olmalı, {factor} verildi")
    return np.clip(np.rint(img.astype(np.float32) * factor), 0, 255).astype(np.uint8)


def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma < 0:
        raise ValueError(f"noise: σ negatif olamaz, {sigma} verildi")
    if sigma == 0:
        return img.copy()
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize == 0:
        return img.copy()
    if ksize < 0 or ksize % 2 == 0:
        raise ValueError(f"blur: çekirdek pozitif tek sayı olmalı, {ksize} verildi")
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def shift(img: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    """İçeriği (dx, dy) piksel kaydırır (pozitif dx sağa, dy aşağı); boşalan kenar 0."""
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = img[
        max(-dy, 0) : h - max(dy, 0), max(-dx, 0) : w - max(dx, 0)
    ]
    return out


def suite_conditions(only: Iterable[str] | None = None) -> list[Condition]:
    names = list(SUITE) if only is None else list(only)
    unknown = [n for n in names if n not in SUITE]
    if unknown:
        raise ValueError(f"bilinmeyen bozulma: {unknown}; seçenekler: {list(SUITE)}")
    conds = [CLEAN]
    for name in SUITE:  # sıra her zaman SUITE sırası
        if name not in names:
            continue
        target, sevs = SUITE[name]
        conds += [Condition(name, float(s), target) for s in sevs if s != IDENTITY.get(name)]
    return conds


def select_conditions(ids: Iterable[str]) -> list[Condition]:
    """Koşul kimliğiyle seçim (ör. test kontrolü: clean, noise-20, misalign-20); SUITE sırası."""
    want = set(ids)
    conds = suite_conditions()
    unknown = sorted(want - {c.id for c in conds})
    if unknown:
        raise ValueError(f"bilinmeyen koşul: {unknown}; seçenekler: {[c.id for c in conds]}")
    return [c for c in conds if c.id in want]


def applies(cond: Condition, mode: str) -> bool:
    needs = MODE_NEEDS[mode]
    if cond.corruption == "clean":
        return True
    if cond.corruption == "misalign":  # tek modalitede kaydırma = etiket kayması
        return needs == {"rgb", "ir"}
    touched = {"rgb", "ir"} if cond.target == "both" else {cond.target}
    return bool(needs & touched)


def image_rng(seed: int, stem: str, cond: Condition) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{stem}|{cond.id}".encode())])


def apply_condition(
    bgr: np.ndarray, ir: np.ndarray, cond: Condition, *, seed: int, stem: str
) -> tuple[np.ndarray, np.ndarray]:
    c, s = cond.corruption, cond.severity
    if c == "clean":
        return bgr.copy(), ir.copy()
    if c == "darken":
        return darken(bgr, s), ir.copy()
    if c == "noise":
        return bgr.copy(), gaussian_noise(ir, s, image_rng(seed, stem, cond))
    if c == "blur":
        return gaussian_blur(bgr, int(s)), gaussian_blur(ir, int(s))
    if c == "misalign":
        return bgr.copy(), shift(ir, int(s), 0)
    if c == "drop_rgb":
        return np.zeros_like(bgr), ir.copy()
    if c == "drop_ir":
        return bgr.copy(), np.zeros_like(ir)
    raise ValueError(f"bilinmeyen bozulma: {c}")
