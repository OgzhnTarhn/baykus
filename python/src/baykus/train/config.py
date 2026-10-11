"""Deney config şeması (spec §10.3). Bilinmeyen anahtar her seviyede ConfigError verir (FR-T2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

MODES = ("rgb", "ir", "early_fusion")
MODALITY_DIR = {"rgb": "rgb", "ir": "ir", "early_fusion": "rgbt4"}
LABEL_SETS = ("ir", "rgb")
SPLITS = ("train", "val", "test")
SLICES = ("illumination", "size", "class")
FIRST_CONV_INITS = ("ul_default", "rgb_mean")
_TOP = {
    "experiment",
    "mode",
    "model",
    "data",
    "label_set",
    "first_conv_init",
    "modality_dropout",
    "train",
    "eval",
    "tracking",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TrainCfg:
    imgsz: int = 640
    epochs: int = 100
    patience: int = 20
    batch: int = 16
    seed: int = 42
    close_mosaic: int = 10
    workers: int = 4
    save_period: int = 5


@dataclass(frozen=True)
class EvalCfg:
    splits: tuple[str, ...] = ("val",)
    label_sets: tuple[str, ...] = LABEL_SETS
    slices: tuple[str, ...] = SLICES
    n_boot: int = 1000


@dataclass(frozen=True)
class DropoutCfg:
    p_rgb: float = 0.0
    p_ir: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    mode: str
    model: str
    data: str
    label_set: str = "ir"
    first_conv_init: str | None = None
    modality_dropout: DropoutCfg | None = None
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    mlflow_experiment: str = "baykus"
    config_hash: str = ""

    @property
    def modality(self) -> str:
        return MODALITY_DIR[self.mode]


def config_hash(raw: dict) -> str:
    blob = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _check_keys(d: object, allowed: set[str], where: str) -> dict:
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: sözlük bekleniyordu")
    extra = sorted(set(d) - allowed)
    if extra:
        raise ConfigError(f"{where}: bilinmeyen anahtar(lar): {', '.join(extra)}")
    return d


def _section(cls: type, d: object, where: str):
    d = _check_keys(d if d is not None else {}, {f.name for f in fields(cls)}, where)
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def _check_subset(values: tuple[str, ...], allowed: tuple[str, ...], where: str) -> None:
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ConfigError(f"{where}: geçersiz değer(ler) {bad}; seçenekler: {allowed}")


def parse_config(raw: dict) -> ExperimentConfig:
    _check_keys(raw, _TOP, "config")
    for key in ("experiment", "mode", "model", "data"):
        if key not in raw:
            raise ConfigError(f"config: zorunlu anahtar eksik: {key}")
    mode = raw["mode"]
    if mode not in MODES:
        raise ConfigError(f"mode: {mode!r} geçersiz; seçenekler: {MODES}")
    label_set = raw.get("label_set", "ir")
    if label_set not in LABEL_SETS:
        raise ConfigError(f"label_set: {label_set!r} geçersiz; seçenekler: {LABEL_SETS}")
    expected = f"data_{MODALITY_DIR[mode]}.yaml"
    if Path(str(raw["data"])).name != expected:
        raise ConfigError(
            f"data: mode={mode} için {expected} bekleniyordu, {raw['data']!r} verildi"
        )

    init, drop_raw = raw.get("first_conv_init"), raw.get("modality_dropout")
    if mode != "early_fusion" and (init is not None or drop_raw is not None):
        raise ConfigError("first_conv_init ve modality_dropout sadece mode=early_fusion için")
    if mode == "early_fusion":
        init = init or "ul_default"
        if init not in FIRST_CONV_INITS:
            raise ConfigError(
                f"first_conv_init: {init!r} geçersiz; seçenekler: {FIRST_CONV_INITS} "
                "('random' H4'te, bütçe varsa)"
            )
    drop = None
    if drop_raw is not None:
        drop = _section(DropoutCfg, drop_raw, "modality_dropout")
        for name in ("p_rgb", "p_ir"):
            p = getattr(drop, name)
            if not 0.0 <= p <= 1.0:
                raise ConfigError(f"modality_dropout.{name}: 0-1 arası olmalı, {p} verildi")
        if drop.p_rgb + drop.p_ir > 1.0:
            raise ConfigError(
                "modality_dropout: p_rgb + p_ir en fazla 1 (aynı örnekte ikisi birden sıfırlanmaz)"
            )

    train = _section(TrainCfg, raw.get("train"), "train")
    ev = _section(EvalCfg, raw.get("eval"), "eval")
    _check_subset(ev.splits, SPLITS, "eval.splits")
    _check_subset(ev.label_sets, LABEL_SETS, "eval.label_sets")
    _check_subset(ev.slices, SLICES, "eval.slices")
    tracking = _check_keys(raw.get("tracking") or {}, {"mlflow_experiment"}, "tracking")

    return ExperimentConfig(
        experiment=str(raw["experiment"]),
        mode=mode,
        model=str(raw["model"]),
        data=str(raw["data"]),
        label_set=label_set,
        first_conv_init=init,
        modality_dropout=drop,
        train=train,
        eval=ev,
        mlflow_experiment=str(tracking.get("mlflow_experiment", "baykus")),
        config_hash=config_hash(raw),
    )


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config bir sözlük (YAML mapping) olmalı")
    return parse_config(raw)
