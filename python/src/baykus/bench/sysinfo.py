"""Bench kaydının sistem bilgisi (spec §14): CPU modeli ve bayrakları, çekirdek, RAM, güç
durumu, yazılım sürümleri, sürecin tepe belleği (FR-X5, NFR-6). Okunamayan alan None."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

import cpuinfo
import psutil

from baykus.runinfo import hardware_info

# INT8'in CPU'da hızlanması bu komut setlerine bağlı (spec §14 INT8 notları)
FLAGS = ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_bf16", "amx_int8", "amx_tile")
PACKAGES = ("onnxruntime", "onnxruntime-gpu", "tensorrt", "numpy", "opencv-python")


def _try(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001 — bilgi alanı; ölçümü durdurmamalı
        return None


def _version(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def peak_rss_mb() -> float | None:
    def read() -> float:
        if sys.platform == "win32":
            return psutil.Process().memory_info().peak_wset / 1e6
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3  # Linux: KB

    return _try(read)


def sysinfo() -> dict:
    info = _try(cpuinfo.get_cpu_info) or {}
    flags = info.get("flags")
    battery = _try(psutil.sensors_battery)
    plugged = None if battery is None else bool(battery.power_plugged)
    if plugged is False:
        print("uyarı: laptop pilde; ölçüm prize takılıyken tekrarlanmalı", file=sys.stderr)
    return {
        **hardware_info(),
        "cpu_brand": info.get("brand_raw"),
        "cpu_flags": None if flags is None else [f for f in FLAGS if f in flags],
        "physical_cores": _try(lambda: psutil.cpu_count(logical=False)),
        "logical_cores": _try(psutil.cpu_count),
        "ram_gb": _try(lambda: round(psutil.virtual_memory().total / 1e9, 1)),
        "power_plugged": plugged,
        "software": {p: _version(p) for p in PACKAGES},
    }
