"""ONNX Runtime statik INT8 nicemleme (H5, spec §14, FR-X3).

QDQ, kanal başına ağırlık, S8S8, MinMax kalibrasyonu. Nicemlenen düğümler Ultralytics 8.4.177'nin
`onnx_int8_quantize` kuralıyla aynı: yalnız Conv/Gemm/MatMul; kutu çözümleme ve TopK gibi
ağırlıksız düğümler FP32 kalır. head_fp32=True: son `/model.N/` altındaki bütün düğümler
(tespit başlığı) de FP32. quant_pre_process kullanılmaz: düğüm adlarını değiştirip baş önekini
bozabilir.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from baykus.edge.frames import load_input, preprocess

WEIGHTED_OPS = frozenset({"Conv", "Gemm", "MatMul"})
_MODEL_RE = re.compile(r"^/model\.(\d+)/")


def head_prefix(model: onnx.ModelProto) -> str:
    idx = [int(m.group(1)) for n in model.graph.node if (m := _MODEL_RE.match(n.name))]
    if not idx:
        raise ValueError("düğüm adlarında /model.N/ yok: Ultralytics ONNX'i değil mi?")
    return f"/model.{max(idx)}/"


def int8_exclusions(model: onnx.ModelProto, head_fp32: bool) -> list[str]:
    names = {n.name for n in model.graph.node if n.op_type not in WEIGHTED_OPS}
    if head_fp32:
        prefix = head_prefix(model)
        names |= {n.name for n in model.graph.node if n.name.startswith(prefix)}
    return sorted(names)


class FrameReader(CalibrationDataReader):
    """Kalibrasyon görüntüleri, bench ve doğrulukla aynı ön işlemeden geçer."""

    def __init__(
        self,
        root: Path,
        split: str,
        stems: Sequence[str],
        modality: str,
        imgsz: int = 640,
        input_name: str = "images",
    ) -> None:
        self.root, self.split, self.stems = Path(root), split, list(stems)
        self.modality, self.imgsz, self.input_name = modality, imgsz, input_name
        self._i = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._i >= len(self.stems):
            return None
        img = load_input(self.root, self.split, self.stems[self._i], self.modality)
        self._i += 1
        return {self.input_name: preprocess(img, self.imgsz)[0]}

    def rewind(self) -> None:
        self._i = 0


class _Counting(CalibrationDataReader):
    def __init__(self, inner: CalibrationDataReader) -> None:
        self.inner, self.count = inner, 0

    def get_next(self):
        x = self.inner.get_next()
        if x is not None:
            self.count += 1
        return x

    def rewind(self) -> None:
        self.inner.rewind()
        self.count = 0


def quantize_int8(src: Path, dst: Path, reader: CalibrationDataReader, *, head_fp32: bool) -> dict:
    src, dst = Path(src), Path(dst)
    model = onnx.load(src)
    exclude = int8_exclusions(model, head_fp32)
    prefix = head_prefix(model)
    counting = _Counting(reader)
    # ModelProto: ORT dosya yolunda şekil çıkarımı yapınca ASCII dışı yolda (Windows) açamıyor
    quantize_static(
        model,
        str(dst),
        counting,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,
    )
    q = onnx.load(dst)  # Ultralytics metadata'sı (names, channels, end2end) korunsun
    del q.metadata_props[:]
    for p in model.metadata_props:
        q.metadata_props.add(key=p.key, value=p.value)
    onnx.save(q, dst)
    return {
        "format": "QDQ",
        "per_channel": True,
        "activation": "int8",
        "weight": "int8",
        "calibrate": "minmax",
        "head_prefix": prefix,
        "head_fp32": head_fp32,
        "excluded_nodes": len(exclude),
        "calibration_images": counting.count,
    }
