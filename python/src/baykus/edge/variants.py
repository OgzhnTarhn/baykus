"""Edge varyant kaydı (H5): ONNX ve TensorRT dışa aktarım biçimleri."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Variant:
    name: str
    fmt: Literal["onnx", "engine"]
    precision: Literal["fp32", "fp16", "int8"]
    head_fp32: bool
    device: Literal["cpu", "cuda"]


VARIANTS: dict[str, Variant] = {
    "onnx_fp32": Variant("onnx_fp32", "onnx", "fp32", False, "cpu"),
    "onnx_fp16": Variant("onnx_fp16", "onnx", "fp16", False, "cpu"),
    "onnx_int8": Variant("onnx_int8", "onnx", "int8", False, "cpu"),
    "onnx_int8_headfp32": Variant("onnx_int8_headfp32", "onnx", "int8", True, "cpu"),
    "trt_fp32": Variant("trt_fp32", "engine", "fp32", False, "cuda"),
    "trt_fp16": Variant("trt_fp16", "engine", "fp16", False, "cuda"),
    "trt_int8": Variant("trt_int8", "engine", "int8", False, "cuda"),
}

ONNX_VARIANTS: tuple[str, ...] = tuple(n for n, v in VARIANTS.items() if v.fmt == "onnx")
TRT_VARIANTS: tuple[str, ...] = tuple(n for n, v in VARIANTS.items() if v.fmt == "engine")

_SUFFIX = {"onnx": ".onnx", "engine": ".engine"}


def model_file(models_dir: Path, run_id: str, variant: str) -> Path:
    v = VARIANTS.get(variant)
    if v is None:
        raise ValueError(f"bilinmeyen varyant: {variant!r}; seçenekler: {tuple(VARIANTS)}")
    return Path(models_dir) / f"{run_id}_{variant}{_SUFFIX[v.fmt]}"
