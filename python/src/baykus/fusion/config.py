"""Geç füzyon config'i (FR-V7). Eğitim config'inden ayrı: model eğitilmez, iki modelin
tahminleri birleşir. Bilinmeyen anahtar ConfigError (eğitim config'iyle aynı kural)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from baykus.train.config import LABEL_SETS, ConfigError, _check_keys, config_hash

_TOP = {"experiment", "mode", "model", "rgb", "ir", "wbf", "eval"}
_WBF = {"iou_thr", "ir_weight", "skip_box_thr", "per_illumination"}
_EVAL = {"label_sets", "n_boot", "seed"}


@dataclass(frozen=True)
class FusionConfig:
    experiment: str
    model: str
    rgb: str
    ir: str
    iou_thr: tuple[float, ...]
    ir_weight: tuple[float, ...]
    skip_box_thr: float = 0.001
    per_illumination: bool = False
    label_sets: tuple[str, ...] = LABEL_SETS
    n_boot: int = 1000
    seed: int = 42
    config_hash: str = ""
    mode: str = "late_fusion"


def parse_fusion_config(raw: dict) -> FusionConfig:
    _check_keys(raw, _TOP, "config")
    for key in ("experiment", "mode", "model", "rgb", "ir", "wbf"):
        if key not in raw:
            raise ConfigError(f"config: zorunlu anahtar eksik: {key}")
    if raw["mode"] != "late_fusion":
        raise ConfigError(f"mode: geç füzyon config'inde 'late_fusion' olmalı, {raw['mode']!r}")
    wbf = _check_keys(raw["wbf"], _WBF, "wbf")
    iou = tuple(float(x) for x in wbf.get("iou_thr") or ())
    irw = tuple(float(x) for x in wbf.get("ir_weight") or ())
    per_ill = bool(wbf.get("per_illumination", False))
    if not irw or any(w <= 0 for w in irw):
        raise ConfigError(f"wbf.ir_weight: boş olmayan pozitif liste olmalı, {irw}")
    if any(not 0 < x < 1 for x in iou):
        raise ConfigError(f"wbf.iou_thr: değerler 0-1 arası olmalı, {iou}")
    if per_ill and iou:
        raise ConfigError("wbf: per_illumination iken iou_thr verilmez; E4'ün seçtiği kullanılır")
    if not per_ill and not iou:
        raise ConfigError("wbf.iou_thr boş olamaz (sadece E4b'de E4'ten gelir)")
    skip = float(wbf.get("skip_box_thr", 0.001))
    if not 0 <= skip < 1:
        raise ConfigError(f"wbf.skip_box_thr: 0-1 arası olmalı, {skip}")
    ev = _check_keys(raw.get("eval") or {}, _EVAL, "eval")
    label_sets = tuple(ev.get("label_sets", LABEL_SETS))
    bad = [x for x in label_sets if x not in LABEL_SETS]
    if bad:
        raise ConfigError(f"eval.label_sets: geçersiz {bad}; seçenekler: {LABEL_SETS}")
    return FusionConfig(
        experiment=str(raw["experiment"]),
        model=str(raw["model"]),
        rgb=str(raw["rgb"]),
        ir=str(raw["ir"]),
        iou_thr=iou,
        ir_weight=irw,
        skip_box_thr=skip,
        per_illumination=per_ill,
        label_sets=label_sets,
        n_boot=int(ev.get("n_boot", 1000)),
        seed=int(ev.get("seed", 42)),
        config_hash=config_hash(raw),
    )


def load_fusion_config(path: Path) -> FusionConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config bir sözlük (YAML mapping) olmalı")
    return parse_fusion_config(raw)
