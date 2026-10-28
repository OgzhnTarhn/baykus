"""Faz 4 parametreleri: `configs/engine/motion.json` (tasarım §5.7; motor da açılışta okur).

Anahtarlar birebir: eksik ya da fazla anahtar `ValueError` (yazım hatası sessizce varsayılana
düşmesin).
`sha256` dosya baytlarının özeti (`/v1/model` `motion_sha256`, sonuç JSON'ları).
`ego.ir_return_frames` IR'ye dönüş kuralının (kanal seçimi) parametresi; ego yalnız IR olduğundan
(tasarım §5.1 bulgusu) şimdilik okunmaz, şema motorla aynı kalsın diye durur.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EgoCfg:
    enabled: bool
    max_corners: int
    quality: float
    min_distance: float
    block_size: int
    lk_win: int
    lk_levels: int
    relink_levels: int
    fb_max_px: float
    ransac_px: float
    ransac_iters: int
    confidence: float
    refine_iters: int
    ok_min_inliers: int
    ok_min_ratio: float
    key_min_inliers: int
    key_min_ratio: float
    key_max_age: int
    key_max_shift_px: float
    mask_dilate_px: float
    mask_min_score: float
    border_px: int
    relink_max_gap: int
    ir_return_frames: int


@dataclass(frozen=True)
class RelocCfg:
    orb_features: int
    candidates: int
    max_dframe: int
    window_frames: int
    every: int
    ransac_px: float
    min_inliers: int
    min_ratio: float
    library_max: int


@dataclass(frozen=True)
class StateCfg:
    k: int
    stationary_max: float
    moving_min: float
    hold: int


@dataclass(frozen=True)
class FrozenCfg:
    mad_max: float
    window: int
    min_frames: int


@dataclass(frozen=True)
class ZonesCfg:
    max_zones: int
    max_points: int
    name_max: int
    dwell_max: int
    enter_obs: int
    exit_obs: int
    gap_max: int
    lost_frames: int
    stopped_obs: int
    dwell_clear_moving: int
    line_side_obs: int
    body_max_bytes: int


@dataclass(frozen=True)
class LogCfg:
    max_events: int


@dataclass(frozen=True)
class SummaryCfg:
    max_segments: int
    ended_keep: int


_SECTIONS: dict[str, type] = {
    "ego": EgoCfg,
    "reloc": RelocCfg,
    "motion": StateCfg,
    "frozen": FrozenCfg,
    "zones": ZonesCfg,
    "log": LogCfg,
    "summary": SummaryCfg,
}


def _section(name: str, cls: type, raw: Any, path: Path) -> Any:
    keys = [f.name for f in fields(cls)]
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: '{name}' nesne olmalı")
    missing, extra = [k for k in keys if k not in raw], sorted(set(raw) - set(keys))
    if missing or extra:
        raise ValueError(f"{path}: '{name}' anahtarları {keys}; eksik {missing}, fazla {extra}")
    return cls(**{k: raw[k] for k in keys})


@dataclass(frozen=True)
class MotionCfg:
    ego: EgoCfg
    reloc: RelocCfg
    motion: StateCfg
    frozen: FrozenCfg
    zones: ZonesCfg
    log: LogCfg
    summary: SummaryCfg
    sha256: str

    @classmethod
    def load(cls, path: Path) -> MotionCfg:
        path = Path(path)
        data = path.read_bytes()
        raw = json.loads(data.decode("utf-8"))
        missing = [k for k in _SECTIONS if k not in raw]
        extra = sorted(set(raw) - set(_SECTIONS))
        if missing or extra:
            raise ValueError(f"{path}: bölümler {list(_SECTIONS)}; eksik {missing}, fazla {extra}")
        parts = {k: _section(k, c, raw[k], path) for k, c in _SECTIONS.items()}
        return cls(**parts, sha256=hashlib.sha256(data).hexdigest())
