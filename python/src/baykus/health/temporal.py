"""Hizasızlığın zaman kuralı (Faz 3 tasarımı §4.2, plan F3 Görev 7).

Gerçek kayma kareden kareye aynı yönde ve büyüklükte kalır, anlamsız kestirim her karede başka
yöne zıplar. Pencere: oturumun son `window` karesi (geçerli dahil). S = penceredeki
`misaligned_frame` kareleri; v̄ = S'nin (dx_px, dy_px) vektörlerinin bileşen bazında ortancası;
C = {i ∈ S : ‖vᵢ − v̄‖₂ ≤ tol_px} (sınır dahil, float64). `misaligned` = |C| ≥ min_frames ve
iki kamera da kayıp değil. Kayıp ve gürültü kuralları kare başına ve önceliklidir.
Oturum yoksa (tek kare) bu sınıf kullanılmaz: `misaligned` = `misaligned_frame`.
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np

from baykus.health.gate import REASONS, decide_gate


class MisalignWindow:
    def __init__(self, window: int = 5, min_frames: int = 3, tol_px: float = 3.0) -> None:
        self.window, self.min_frames, self.tol_px = window, min_frames, float(tol_px)
        self._buf: deque[tuple[bool, float | None, float | None]] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, h: dict) -> dict:
        """Tek kare sağlık sözlüğü (`assess` çıktısı) → zaman kuralı uygulanmış kopyası."""
        out = copy.deepcopy(h)
        pair = out["pair"]
        flagged = bool(pair["misaligned_frame"])
        if flagged and (pair.get("dx_px") is None or pair.get("dy_px") is None):
            raise ValueError("işaretli karede kayma vektörü (dx_px, dy_px) yok")
        self._buf.append((flagged, pair.get("dx_px"), pair.get("dy_px")))
        s = np.array([(dx, dy) for f, dx, dy in self._buf if f], dtype=np.float64)
        n_c = 0
        if len(s):
            center = np.median(s, axis=0)
            n_c = int(np.sum(np.linalg.norm(s - center, axis=1) <= self.tol_px))
        rgb_lost = "lost" in out["rgb"]["flags"]
        ir_lost = "lost" in out["ir"]["flags"]
        ir_noisy = "noisy" in out["ir"]["flags"]
        pair["misaligned"] = n_c >= self.min_frames and not (rgb_lost or ir_lost)
        out["gate"], code = decide_gate(rgb_lost, ir_lost, ir_noisy, pair["misaligned"])
        out["reason_code"] = code
        out["reason"] = REASONS[code] if code else ""
        return out

    def sync_pairs(self, healths: list[dict]) -> list[dict]:
        return [self.update(h) for h in healths]
