"""Çalıştırma kimliği ve izlenebilirlik: git commit, run_id, yazılım sürümleri, JSON (FR-T5, S1)."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SOFTWARE = ("torch", "ultralytics", "numpy", "pycocotools")
_MODEL_RE = re.compile(r"(yolo\d+)([nsmlx])")


def git_info(cwd: Path | None = None) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _parse_model(model: str) -> re.Match:
    m = _MODEL_RE.fullmatch(Path(model).stem)
    if m is None:
        raise ValueError(f"model adı çözümlenemedi (ör. yolo26s.pt bekleniyordu): {model}")
    return m


def model_family(model: str) -> str:
    return _parse_model(model).group(1)


def model_size(model: str) -> str:
    return _parse_model(model).group(2)


def make_run_id(experiment: str, model: str, commit: str | None, when: dt.datetime) -> str:
    short = (commit or "nogit")[:7]
    return f"{when:%Y-%m-%d}_{experiment.replace('_', '-')}_{model_size(model)}_{short}"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, data: dict) -> Path:
    """Atomik yazma: önce .tmp, sonra os.replace. Colab oturumu yazma anında koparsa
    Drive'da yarım JSON kalmaz (sağlamlık paketi var olan dosyayı "bitti" sayar)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return path


def hardware_info() -> dict:
    """GPU adı ve CPU çekirdek sayısı: aynı deney T4/L4/A100'de farklı sürer (gpu-budget.md)."""
    out: dict = {
        "device": platform.processor() or platform.machine() or "cpu",
        "driver": None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            out.update(device=torch.cuda.get_device_name(0), driver=torch.version.cuda)
    except ImportError:
        pass
    return out


def software_versions() -> dict:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for pkg in SOFTWARE:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out
