"""Model dizini (`models/`): ONNX + eşikler + kalibrasyon tablosu + kapı modeli.

Manifest (`configs/engine/models.json`) verilirse her dosyanın sha256'sı denetlenir; motor da
açılışta aynı denetimi yapar. Kapı modeli başka bir eşik dosyasından üretildiyse hata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baykus.health.gate import Thresholds, load_thresholds
from baykus.health.guard import GateModel, load_gate_model

FILES = ("e3c.onnx", "thresholds.json", "calibration.json", "gate_model.json")


@dataclass(frozen=True)
class Models:
    onnx: Path
    thresholds: Thresholds
    calib_x: np.ndarray
    calib_y: np.ndarray
    gate: GateModel


def check_manifest(models_dir: Path, manifest: Path) -> None:
    files = json.loads(Path(manifest).read_text(encoding="utf-8"))["files"]
    for name, rec in files.items():
        p = Path(models_dir) / name
        if not p.is_file():
            raise FileNotFoundError(f"{p} yok")
        if hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
            raise ValueError(f"{name}: sha256 manifestle uyuşmuyor ({manifest})")


def load_models(models_dir: Path, manifest: Path | None = None) -> Models:
    d = Path(models_dir)
    missing = [n for n in FILES if not (d / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{d}: eksik model dosyaları {missing}")
    if manifest is not None:
        check_manifest(d, manifest)
    th = load_thresholds(d / "thresholds.json")
    gate = load_gate_model(d / "gate_model.json")
    if gate.thresholds_sha256 != th.sha256:
        raise ValueError("gate_model.json başka bir thresholds.json'dan üretilmiş (sha256 farklı)")
    table = json.loads((d / "calibration.json").read_text(encoding="utf-8"))["table"]
    return Models(
        onnx=d / "e3c.onnx",
        thresholds=th,
        calib_x=np.asarray(table["x"], np.float64),
        calib_y=np.asarray(table["y"], np.float64),
        gate=gate,
    )
