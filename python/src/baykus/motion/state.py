"""Hareket durumu, Python referansı (Faz 4 tasarımı §5.2; motor `engine/src/motion.cpp`).

İz başına yalnız `ego_ok` karelerde harita merkezi `cₜ = Mₜ(kutu merkezi)` ve harita boyu
`sₜ = √(w·h)·σ(Mₜ)`. Geçerli karede ve tam `t − K` karesinde kaydı olan, `(t − K, t]` aralığındaki
bütün karelerde `ego_ok` ve aynı nesilde olan iz için `v = ‖cₜ − cₜ₋ₖ‖ / K / sₜ`.
Karar: `v ≤ stationary_max` duran, `v ≥ moving_min` hareketli (sınırlar dahil), arada önceki karar.
`v` yoksa önceki karar son `v`'den bu yana en çok `hold` kare tutulur, sonra `unknown`.
Yeni nesilde (birleşmesiz) geçmiş ve kararlar silinir; birleşmede geçmiş `T` ile hedef nesle
taşınır.
`t` = çağrı sayacı (oturum içi istek sırası `n`).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from baykus.motion.config import MotionCfg
from baykus.motion.ego import EgoResult
from baykus.motion.geometry import apply, scale
from baykus.motion.images import Box


class TrackObs(NamedTuple):
    track_id: int
    box: Box


class TrackMotion(NamedTuple):
    motion: str  # moving | stationary | unknown
    speed_rel: float | None
    map_xy: tuple[float, float] | None


@dataclass
class _Track:
    hist: dict[int, tuple[int, np.ndarray, float]] = field(default_factory=dict)  # n → (gen, c, s)
    decision: str = "unknown"
    last_v_n: int | None = None
    last_seen: int = 0


class MotionState:
    def __init__(self, cfg: MotionCfg):
        self.cfg = cfg.motion
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self._gen: int | None = None  # son ok karenin nesli
        self._ego: deque[tuple[int, bool, int]] = deque(maxlen=self.cfg.k)  # (n, ok, gen)
        self._tracks: dict[int, _Track] = {}

    def _relabel(self, from_gen: int, to_gen: int, t: np.ndarray) -> None:
        sig = scale(t)
        for tr in self._tracks.values():
            for n, (g, c, s) in list(tr.hist.items()):
                if g == from_gen:
                    tr.hist[n] = (to_gen, apply(t, c[None])[0], sig * s)
        self._ego = deque(
            ((n, ok, to_gen if g == from_gen else g) for n, ok, g in self._ego), maxlen=self.cfg.k
        )
        if self._gen == from_gen:
            self._gen = to_gen

    def _window_ok(self, gen: int) -> bool:
        """`(t − K, t]` bütün kareleri ok ve aynı nesilde (deque son K kareyi tutar)."""
        k = self.cfg.k
        return len(self._ego) == k and all(ok and g == gen for _, ok, g in self._ego)

    def update(self, ego: EgoResult, tracks: list[TrackObs]) -> dict[int, TrackMotion]:
        cfg = self.cfg
        self.n += 1
        n = self.n
        if ego.ok and ego.merge is not None:
            self._relabel(ego.merge.from_gen, ego.merge.to_gen, ego.merge.transform)
        if ego.ok and self._gen is not None and ego.map_gen != self._gen:
            self._tracks.clear()  # yeni nesil: geçmiş ve kararlar silinir
            self._ego.clear()
        if ego.ok:
            self._gen = ego.map_gen
        self._ego.append((n, ego.ok, ego.map_gen))
        window_ok = ego.ok and self._window_ok(ego.map_gen)

        out: dict[int, TrackMotion] = {}
        for obs in tracks:
            tr = self._tracks.setdefault(obs.track_id, _Track())
            tr.last_seen = n
            prev = tr.decision
            if tr.last_v_n is None or n - 1 - tr.last_v_n > cfg.hold:
                prev = "unknown"
            v = c = None
            if ego.ok:
                m = ego.frame_to_map
                x1, y1, x2, y2 = obs.box
                c = apply(m, np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))[0]
                s = math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)) * scale(m)
                tr.hist[n] = (ego.map_gen, c, s)
                old = tr.hist.get(n - cfg.k)
                if window_ok and old is not None and old[0] == ego.map_gen and s > 0:
                    v = float(np.linalg.norm(c - old[1])) / cfg.k / s
            if v is not None:
                if v <= cfg.stationary_max:
                    tr.decision = "stationary"
                elif v >= cfg.moving_min:
                    tr.decision = "moving"
                else:
                    tr.decision = prev
                tr.last_v_n = n
            elif tr.last_v_n is None or n - tr.last_v_n > cfg.hold:
                tr.decision = "unknown"
            for old_n in [h for h in tr.hist if h < n - cfg.k]:
                del tr.hist[old_n]
            xy = None if c is None else (float(c[0]), float(c[1]))
            out[obs.track_id] = TrackMotion(tr.decision, v, xy)
        self._prune(n)
        return out

    def _prune(self, n: int) -> None:
        """`K + hold` karedir görünmeyen iz: geçmişi kullanılamaz, kararı süresi dolmuş."""
        horizon = self.cfg.k + self.cfg.hold
        for tid in [t for t, tr in self._tracks.items() if n - tr.last_seen > horizon]:
            del self._tracks[tid]
