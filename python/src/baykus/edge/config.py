"""Edge config'i (H5): dışa aktarım, nicemleme, bench ve sevk kuralı. Bilinmeyen anahtar,
bilinmeyen varyant ya da bilinmeyen modele başvuran pipeline ConfigError verir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from baykus.edge.variants import VARIANTS
from baykus.train.config import ConfigError, _check_keys, config_hash

_TOP = {
    "experiment",
    "imgsz",
    "seed",
    "calibration",
    "variants",
    "models",
    "pipelines",
    "bench",
    "rule",
}
_CALIB = {"split", "n"}
_MODEL = {"run_id", "config"}
_PIPELINE = {"kind", "model", "rgb", "ir", "selection"}
_BENCH = {"warmup", "runs", "images", "conf"}
_RULE = {"target_p95_ms", "max_map50_drop", "order"}
_KINDS = ("single", "late_fusion")


@dataclass(frozen=True)
class Calib:
    split: str
    n: int


@dataclass(frozen=True)
class ModelRef:
    run_id: str
    config: str


@dataclass(frozen=True)
class Pipeline:
    kind: str
    model: str | None = None
    rgb: str | None = None
    ir: str | None = None
    selection: str | None = None


@dataclass(frozen=True)
class BenchCfg:
    warmup: int
    runs: int
    images: int
    conf: float


@dataclass(frozen=True)
class Rule:
    target_p95_ms: float
    max_map50_drop: float
    order: tuple[str, ...]


@dataclass(frozen=True)
class EdgeConfig:
    experiment: str
    imgsz: int
    seed: int
    calibration: Calib
    variants: tuple[str, ...]
    models: dict[str, ModelRef]
    pipelines: dict[str, Pipeline]
    bench: BenchCfg
    rule: Rule
    config_hash: str = ""


def _required(d: dict, keys: tuple[str, ...], where: str) -> None:
    for key in keys:
        if key not in d:
            raise ConfigError(f"{where}: zorunlu anahtar eksik: {key}")


def _parse_models(raw: object) -> dict[str, ModelRef]:
    if not isinstance(raw, dict):
        raise ConfigError("models: sözlük bekleniyordu")
    models: dict[str, ModelRef] = {}
    for name, ref in raw.items():
        ref = _check_keys(ref, _MODEL, f"models.{name}")
        _required(ref, ("run_id", "config"), f"models.{name}")
        models[str(name)] = ModelRef(run_id=str(ref["run_id"]), config=str(ref["config"]))
    return models


def _parse_pipelines(raw: object, models: dict[str, ModelRef]) -> dict[str, Pipeline]:
    if not isinstance(raw, dict):
        raise ConfigError("pipelines: sözlük bekleniyordu")
    pipelines: dict[str, Pipeline] = {}
    for name, p in raw.items():
        p = _check_keys(p, _PIPELINE, f"pipelines.{name}")
        _required(p, ("kind",), f"pipelines.{name}")
        kind = p["kind"]
        if kind not in _KINDS:
            raise ConfigError(f"pipelines.{name}.kind: {kind!r} geçersiz; seçenekler: {_KINDS}")
        if kind == "single":
            model = p.get("model")
            if model is None:
                raise ConfigError(f"pipelines.{name}: kind=single için model zorunlu")
            if model not in models:
                raise ConfigError(f"pipelines.{name}.model: bilinmeyen model {model!r}")
            pipelines[str(name)] = Pipeline(kind=kind, model=str(model))
        else:
            _required(p, ("rgb", "ir", "selection"), f"pipelines.{name}")
            for key in ("rgb", "ir"):
                if p[key] not in models:
                    raise ConfigError(f"pipelines.{name}.{key}: bilinmeyen model {p[key]!r}")
            pipelines[str(name)] = Pipeline(
                kind=kind, rgb=str(p["rgb"]), ir=str(p["ir"]), selection=str(p["selection"])
            )
    return pipelines


def parse_edge_config(raw: dict) -> EdgeConfig:
    _check_keys(raw, _TOP, "config")
    _required(
        raw,
        (
            "experiment",
            "imgsz",
            "seed",
            "calibration",
            "variants",
            "models",
            "pipelines",
            "bench",
            "rule",
        ),
        "config",
    )

    calib = _check_keys(raw["calibration"], _CALIB, "calibration")
    _required(calib, ("split", "n"), "calibration")
    calibration = Calib(split=str(calib["split"]), n=int(calib["n"]))

    variants = tuple(str(v) for v in raw["variants"])
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise ConfigError(f"variants: bilinmeyen varyant(lar) {bad}; seçenekler: {tuple(VARIANTS)}")

    models = _parse_models(raw["models"])
    pipelines = _parse_pipelines(raw["pipelines"], models)

    bench_raw = _check_keys(raw["bench"], _BENCH, "bench")
    _required(bench_raw, ("warmup", "runs", "images", "conf"), "bench")
    bench = BenchCfg(
        warmup=int(bench_raw["warmup"]),
        runs=int(bench_raw["runs"]),
        images=int(bench_raw["images"]),
        conf=float(bench_raw["conf"]),
    )

    rule_raw = _check_keys(raw["rule"], _RULE, "rule")
    _required(rule_raw, ("target_p95_ms", "max_map50_drop", "order"), "rule")
    order = tuple(str(x) for x in rule_raw["order"])
    bad_order = [x for x in order if x not in pipelines]
    if bad_order:
        raise ConfigError(
            f"rule.order: bilinmeyen pipeline(lar) {bad_order}; seçenekler: {tuple(pipelines)}"
        )
    rule = Rule(
        target_p95_ms=float(rule_raw["target_p95_ms"]),
        max_map50_drop=float(rule_raw["max_map50_drop"]),
        order=order,
    )

    return EdgeConfig(
        experiment=str(raw["experiment"]),
        imgsz=int(raw["imgsz"]),
        seed=int(raw["seed"]),
        calibration=calibration,
        variants=variants,
        models=models,
        pipelines=pipelines,
        bench=bench,
        rule=rule,
        config_hash=config_hash(raw),
    )


def load_edge_config(path: Path) -> EdgeConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config bir sözlük (YAML mapping) olmalı")
    return parse_edge_config(raw)


def repo_root(path: Path) -> Path:
    """Config yolundan yukarı doğru 'configs' klasörünü barındıran ilk dizin (E9)."""
    for parent in Path(path).resolve().parents:
        if (parent / "configs").is_dir():
            return parent
    raise ConfigError(f"{path}: 'configs' klasörü içeren üst dizin bulunamadı")
