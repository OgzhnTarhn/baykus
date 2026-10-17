"""Kare istatistikleri (Faz 2 tasarımı §4.1, plan F2 Spec yorumları 7).

CPU'da hızlı, eğitimsiz ölçüler. RGB bellekte BGR uint8 (H, W, 3), IR gri uint8 (H, W).
Hesaplanamayan değer None (asla NaN): sonuçlar doğrudan JSON'a yazılır.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from baykus.data.illumination import mean_brightness

FEATURES = (
    "rgb_std",
    "rgb_noise",
    "rgb_blur",
    "rgb_brightness",
    "ir_std",
    "ir_noise",
    "ir_blur",
    "misalign_px",
    "misalign_response",
)

# Immerkær (1996): iki Laplace farkı; doğrusal eğimde ve sabit görüntüde sıfır yanıt
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def pixel_std(gray: np.ndarray) -> float:
    return float(gray.std())


def noise_sigma(gray: np.ndarray) -> float:
    """Gauss gürültüsünün σ kestirimi: √(π/2) / (6 (W−2)(H−2)) · Σ |I ∗ M| (kenarlar hariç)."""
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    resp = cv2.filter2D(gray.astype(np.float64), -1, _IMMERKAER)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) * np.abs(resp).sum() / (6 * (w - 2) * (h - 2)))


def blur_score(gray: np.ndarray) -> float:
    """Laplace varyansı: düşükse görüntü bulanık."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _edges(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(
        gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def misalign(bgr: np.ndarray, ir: np.ndarray) -> tuple[float | None, float | None]:
    """RGB ve IR kenar haritaları arasındaki kayma (tam çözünürlük pikseli) ve faz korelasyonu
    tepe yanıtı. Kenar büyüklüğü kullanıldığı için termal kutup tersi sonucu değiştirmez.
    Kameralardan biri sabitse (kayıp) (None, None)."""
    a, b = _edges(_gray(bgr)), _edges(ir)
    if a.std() == 0 or b.std() == 0:
        return None, None
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    px, r = 2.0 * math.hypot(dx, dy), float(response)
    if not (math.isfinite(px) and math.isfinite(r)):
        return None, None
    return float(px), r


def frame_features(bgr: np.ndarray, ir: np.ndarray) -> dict[str, float | None]:
    g = _gray(bgr)
    px, response = misalign(bgr, ir)
    return {
        "rgb_std": pixel_std(g),
        "rgb_noise": noise_sigma(g),
        "rgb_blur": blur_score(g),
        "rgb_brightness": float(mean_brightness(bgr)),
        "ir_std": pixel_std(ir),
        "ir_noise": noise_sigma(ir),
        "ir_blur": blur_score(ir),
        "misalign_px": px,
        "misalign_response": response,
    }
