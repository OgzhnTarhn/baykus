"""ONNX ve TensorRT dışa aktarımı (H5): manifest'li, yeniden başlatılabilir.

Ultralytics dosyayı ağırlığın yanına yazar (best.onnx / best.engine); her çağrıdan hemen sonra
`model_file(out_dir, run_id, variant)`'a taşınır. Manifest her varyanttan sonra yazılır: Colab
oturumu koparsa kalan varyantlar yeniden üretilir, bitmişler atlanır.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

from baykus.edge.quantize import FrameReader, quantize_int8
from baykus.edge.variants import VARIANTS, model_file
from baykus.runinfo import write_json

_CHUNK = 1 << 20
_INT8_ONNX = ("onnx_int8", "onnx_int8_headfp32")
_TRT_QUANT = {"fp32": 32, "fp16": 16, "int8": 8}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def onnx_input(path: Path) -> dict:
    """İlk girişin şekli ve eleman tipi (ONNX Runtime'ın beklediği imza)."""
    model = onnx.load(str(path))
    inp = model.graph.input[0]
    dims = [int(d.dim_value) for d in inp.type.tensor_type.shape.dim]
    dtype = helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
    return {"shape": dims, "dtype": np.dtype(dtype).name}


def _ultralytics_export(weights: Path, variant: str, imgsz: int, calib_yaml: Path | None) -> str:
    from ultralytics import YOLO

    v = VARIANTS[variant]
    kw: dict = {"imgsz": imgsz, "nms": False, "batch": 1, "dynamic": False}
    if v.fmt == "onnx":
        kw.update(format="onnx", simplify=True, device="cpu")
        if v.precision == "fp16":
            kw["quantize"] = 16
    else:
        kw.update(format="engine", device=0, quantize=_TRT_QUANT[v.precision])
        if v.precision == "int8":
            kw.update(data=str(calib_yaml), split="train", fraction=1.0)
    return YOLO(str(weights)).export(**kw)


def export_model(
    weights: Path,
    out_dir: Path,
    run_id: str,
    modality: str,
    variants: Sequence[str],
    *,
    imgsz: int,
    root: Path | None = None,
    stems: Sequence[str] = (),
    calib_yaml: Path | None = None,
) -> dict[str, dict]:
    weights, out_dir = Path(weights), Path(out_dir)
    wanted = list(variants)
    bad = [v for v in wanted if v not in VARIANTS]
    if bad:
        raise ValueError(f"bilinmeyen varyant(lar) {bad}; seçenekler: {tuple(VARIANTS)}")
    if any(v in _INT8_ONNX for v in wanted) and (root is None or not stems):
        raise ValueError("INT8 ONNX için kalibrasyon görüntüleri gerekli (root ve stems)")
    if "trt_int8" in wanted and calib_yaml is None:
        raise ValueError("TensorRT INT8 için kalibrasyon YAML'ı gerekli (calib_yaml)")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{run_id}_manifest.json"
    manifest: dict[str, dict] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )

    order = list(wanted)
    if any(v in _INT8_ONNX for v in wanted) and "onnx_fp32" not in order:
        order.append("onnx_fp32")  # INT8'in girdisi; istenmese de üretilir
    order = [v for v in VARIANTS if v in order]

    for v in order:
        target = model_file(out_dir, run_id, v)
        if v in manifest and target.exists():
            continue
        t0 = time.monotonic()
        quant = None
        if v in _INT8_ONNX:
            quant = quantize_int8(
                model_file(out_dir, run_id, "onnx_fp32"),
                target,
                FrameReader(root, "train", stems, modality, imgsz),
                head_fp32=VARIANTS[v].head_fp32,
            )
        else:
            produced = _ultralytics_export(weights, v, imgsz, calib_yaml)
            shutil.move(str(produced), str(target))
        manifest[v] = {
            "variant": v,
            "file": str(target),
            "size_mb": round(target.stat().st_size / 1e6, 3),
            "sha256": sha256_file(target),
            "seconds": round(time.monotonic() - t0, 1),
            "weights": str(weights),
            "input": onnx_input(target) if VARIANTS[v].fmt == "onnx" else None,
            "quant": quant,
        }
        write_json(manifest_path, manifest)
        print(v, target, manifest[v]["size_mb"], flush=True)
    return {v: manifest[v] for v in wanted}
