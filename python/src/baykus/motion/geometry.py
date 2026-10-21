"""2×3 benzerlik / afin dönüşümleri (Faz 4 tasarımı §5.1, §5.11).

Yarı çözünürlük piksel `i` tam çözünürlükte `2i + 0,5`: `S = [[2,0,0.5],[0,2,0.5]]`,
`A_full = S·A·S⁻¹`. `compose(a, b)` = önce `b` sonra `a` (a∘b).
"""

from __future__ import annotations

import math

import numpy as np

_S = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
_S_INV = np.linalg.inv(_S)

# Kabul 3 sentetik dönüşümleri (tasarım §5.9 sabit liste): (tx, ty, dönme °, ölçek), tam çözünürlük,
# görüntü merkezi etrafında; öteleme ±20 px, dönme ±1°, ölçek ±%3.
SIMILARITIES: tuple[tuple[float, float, float, float], ...] = (
    (20.0, 0.0, 0.0, 1.0),
    (-12.0, 15.0, 0.5, 1.0),
    (5.0, -20.0, -1.0, 1.03),
    (-20.0, -8.0, 1.0, 0.97),
    (14.0, 10.0, -0.6, 1.02),
)


def _h(a: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(a, np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])


def to_full(a_half: np.ndarray) -> np.ndarray:
    return (_S @ _h(a_half) @ _S_INV)[:2]


def invert(a: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_h(a))[:2]


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (_h(a) @ _h(b))[:2]


def apply(a: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    a = np.asarray(a, np.float64).reshape(2, 3)
    return p @ a[:, :2].T + a[:, 2]


def scale(a: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(2, 3)
    return math.sqrt(abs(a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]))


def corner_error(a: np.ndarray, b: np.ndarray, w: float, h: float) -> float:
    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    return float(np.max(np.linalg.norm(apply(a, corners) - apply(b, corners), axis=1)))


def similarity(
    tx: float, ty: float, deg: float, s: float, center: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """`center` etrafında `deg` derece dönme ve `s` ölçek, sonra (tx, ty) öteleme."""
    c, k = s * math.cos(math.radians(deg)), s * math.sin(math.radians(deg))
    cx, cy = center
    return np.array(
        [[c, -k, cx - c * cx + k * cy + tx], [k, c, cy - k * cx - c * cy + ty]], np.float64
    )
