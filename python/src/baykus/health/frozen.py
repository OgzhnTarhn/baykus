"""Donmuş kare (Faz 4 tasarımı §5.6; motor `health`).

Kanal başına, oturumlu istekte: yarı çözünürlük gri görüntünün önceki kareyle ortalama mutlak farkı
`diff_mad` (0-255); `frozen` = `diff_mad ≤ mad_max` (sınır dahil). Kayıp kanal (`lost`) donmuş
sayılmaz. İlk kare, görüntüsü olmayan kare ya da boyutu değişen kare: `diff_mad = None`.
`alarm` = son `window` karenin (geçerli dahil) en az `min_frames`'i donmuş (Faz 3 alarm sözleşmesi).
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from baykus.motion.config import MotionCfg

CHANNELS = ("rgb", "ir")


class FrozenDetector:
    def __init__(self, cfg: MotionCfg):
        self.cfg = cfg.frozen
        self.reset()

    def reset(self) -> None:
        self._prev: dict[str, np.ndarray | None] = {ch: None for ch in CHANNELS}
        self._buf: dict[str, deque[bool]] = {ch: deque(maxlen=self.cfg.window) for ch in CHANNELS}

    def _channel(self, ch: str, img: np.ndarray | None, lost: bool) -> dict:
        prev, self._prev[ch] = self._prev[ch], img
        mad = None
        if img is not None and prev is not None and prev.shape == img.shape:
            mad = float(cv2.absdiff(img, prev).mean())
        frozen = mad is not None and not lost and mad <= self.cfg.mad_max
        self._buf[ch].append(frozen)
        return {
            "frozen": frozen,
            "diff_mad": mad,
            "alarm": sum(self._buf[ch]) >= self.cfg.min_frames,
        }

    def update(
        self,
        rgb_half: np.ndarray | None,
        ir_half: np.ndarray | None,
        rgb_lost: bool,
        ir_lost: bool,
    ) -> dict:
        return {
            "rgb": self._channel("rgb", rgb_half, rgb_lost),
            "ir": self._channel("ir", ir_half, ir_lost),
        }
