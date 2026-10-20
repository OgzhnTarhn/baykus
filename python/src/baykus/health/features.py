"""Kare istatistikleri (Faz 2 tasarımı §4.1, plan F2 Spec yorumları 7).

CPU'da hızlı, eğitimsiz ölçüler. RGB bellekte BGR uint8 (H, W, 3), IR gri uint8 (H, W).
Hesaplanamayan değer None (asla NaN): sonuçlar doğrudan JSON'a yazılır.
"""

from __future__ import annotations

import math
from functools import lru_cache

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
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# OpenCV istatistikleri numpy'dan ≈ 5-10 kat hızlı (tasarım §4.1: hedef < 5 ms / kare)
def pixel_std(gray: np.ndarray) -> float:
    return float(cv2.meanStdDev(gray)[1][0, 0])


def noise_sigma(gray: np.ndarray) -> float:
    """Gauss gürültüsünün σ kestirimi: √(π/2) / (6 (W−2)(H−2)) · Σ |I ∗ M| (kenarlar hariç)."""
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    resp = cv2.filter2D(gray, cv2.CV_32F, _IMMERKAER)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) * cv2.norm(resp, cv2.NORM_L1) / (6 * (w - 2) * (h - 2)))


def blur_score(gray: np.ndarray) -> float:
    """Laplace varyansı: düşükse görüntü bulanık."""
    return float(cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_32F))[1][0, 0] ** 2)


def _edges(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(
        gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA
    ).astype(np.float64)
    gx = cv2.Sobel(small, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(small, cv2.CV_64F, 0, 1)
    return cv2.magnitude(gx, gy)


@lru_cache(maxsize=4)
def _window(w: int, h: int) -> np.ndarray:
    return cv2.createHanningWindow((w, h), cv2.CV_64F)


def misalign_vector(bgr: np.ndarray, ir: np.ndarray) -> tuple[float, float, float] | None:
    """IR'nin RGB'ye göre kayma vektörü (dx sağa, dy aşağı; tam çözünürlük pikseli = yarı
    çözünürlük × 2) ve faz korelasyonu tepe yanıtı. Kenar büyüklüğü kullanıldığı için termal
    kutup tersi sonucu değiştirmez. Kameralardan biri sabitse (kayıp) ya da yarı çözünürlükte bir
    kenar 2 px'ten kısaysa None."""
    if min(ir.shape[0], ir.shape[1]) // 2 < 2:
        return None
    a, b = _edges(_gray(bgr)), _edges(ir)
    if pixel_std(a) == 0 or pixel_std(b) == 0:
        return None
    (dx, dy), response = cv2.phaseCorrelate(a, b, _window(a.shape[1], a.shape[0]))
    dx, dy, r = 2.0 * float(dx), 2.0 * float(dy), float(response)
    if not all(math.isfinite(v) for v in (dx, dy, r)):
        return None
    return dx, dy, r


def misalign(bgr: np.ndarray, ir: np.ndarray) -> tuple[float | None, float | None]:
    """Kayma büyüklüğü ‖(dx, dy)‖ ve yanıt; kayıpta (None, None)."""
    v = misalign_vector(bgr, ir)
    if v is None:
        return None, None
    return math.hypot(v[0], v[1]), v[2]


def frame_features(
    bgr: np.ndarray, ir: np.ndarray, vector: bool = False
) -> dict[str, float | None]:
    """`FEATURES` sözlüğü; `vector=True` iken ayrıca `misalign_dx`, `misalign_dy` (Faz 3 zaman
    kuralı). `train_features.csv` biçimi değişmez (varsayılan False)."""
    g = _gray(bgr)
    v = misalign_vector(bgr, ir)
    px, response = (None, None) if v is None else (math.hypot(v[0], v[1]), v[2])
    f = {
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
    if vector:
        f["misalign_dx"], f["misalign_dy"] = (None, None) if v is None else (v[0], v[1])
    return f
