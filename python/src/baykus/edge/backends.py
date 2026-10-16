"""ONNX Runtime ve TensorRT arka uçları (H5, FR-X4).

OrtBackend — ONNX Runtime oturumu; giriş tipi/şekli modelden okunur.
TrtBackend — Ultralytics AutoBackend ile TensorRT motoru.
open_backend — dosya uzantısına göre arka uç seçer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import numpy as np


class Backend(Protocol):
    input_dtype: np.dtype
    channels: int
    imgsz: int
    provider: str
    threads: int | None

    def infer(self, x: np.ndarray) -> np.ndarray: ...


class OrtBackend:
    """ONNX Runtime oturumu; giriş tipi float16 ise girdi float16'ya çevrilir."""

    def __init__(
        self, path: Path, threads: int | None = None, provider: Literal["cpu", "cuda"] = "cpu"
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads is not None:
            opts.intra_op_num_threads = threads
        providers = (
            ["CPUExecutionProvider"]
            if provider == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(path), opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_dtype = np.dtype(np.float16 if inp.type == "tensor(float16)" else np.float32)
        self.channels = int(inp.shape[1])
        self.imgsz = int(inp.shape[2])
        self.input_name = inp.name
        self.threads = threads
        self.provider = provider

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]


class TrtBackend:
    """Ultralytics AutoBackend ile TensorRT motoru (cuda:0)."""

    def __init__(self, path: Path) -> None:
        import torch
        from ultralytics.nn.autobackend import AutoBackend

        self.device = torch.device("cuda:0")
        self.model = AutoBackend(str(path), device=self.device)
        self.fp16 = bool(getattr(self.model, "fp16", False))
        self.input_dtype = np.dtype(np.float16 if self.fp16 else np.float32)
        meta = getattr(self.model, "metadata", {}) or {}
        self.imgsz = int(meta["imgsz"][0]) if "imgsz" in meta else 640
        self.channels = int(getattr(self.model, "ch", 3))
        self.provider = "tensorrt"
        self.threads = None

    def infer(self, x: np.ndarray) -> np.ndarray:
        import torch

        t = torch.from_numpy(x).to(self.device)
        if self.fp16:
            t = t.half()
        y = self.model(t)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y.float().cpu().numpy()


def open_backend(path: Path, threads: int | None = None) -> Backend:
    path = Path(path)
    if path.suffix == ".onnx":
        return OrtBackend(path, threads=threads)
    if path.suffix == ".engine":
        return TrtBackend(path)
    raise ValueError(f"bilinmeyen arka uç dosyası: {path}")
