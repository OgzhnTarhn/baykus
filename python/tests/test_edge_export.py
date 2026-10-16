import json

import pytest

from baykus.edge.export import export_model
from tests.fake_ultralytics import install, make_weights


def test_export_calls_and_names(tmp_path, monkeypatch):
    fake = install(monkeypatch)
    w = make_weights(tmp_path / "runs" / "R", "R")
    out = export_model(
        w,
        tmp_path / "m",
        "R",
        "rgb",
        ["onnx_fp32", "onnx_fp16", "trt_int8"],
        imgsz=640,
        calib_yaml=tmp_path / "c.yaml",
    )
    calls = {c.get("quantize"): c for c in fake.export_calls}
    assert calls[None]["format"] == "onnx"
    assert calls[None]["nms"] is False
    assert calls[None]["device"] == "cpu"
    assert calls[None]["batch"] == 1
    assert calls[None]["dynamic"] is False
    assert calls[None]["imgsz"] == 640
    assert calls[16]["format"] == "onnx"
    assert calls[16]["device"] == "cpu"
    assert calls[8]["format"] == "engine"
    assert calls[8]["split"] == "train"
    assert str(calls[8]["data"]).endswith("c.yaml")
    assert calls[8]["device"] == 0
    assert calls[8]["nms"] is False
    assert (tmp_path / "m/R_trt_int8.engine").exists()
    assert (tmp_path / "m/R_onnx_fp32.onnx").exists()
    assert set(out) == {"onnx_fp32", "onnx_fp16", "trt_int8"}
    assert out["onnx_fp32"]["input"] == {"shape": [1, 3, 8, 8], "dtype": "float32"}
    assert len(out["onnx_fp32"]["sha256"]) == 64
    assert out["trt_int8"]["input"] is None
    manifest = json.loads((tmp_path / "m/R_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == set(out)
    assert not (tmp_path / "runs/R/weights/best.onnx").exists()
    assert not (tmp_path / "runs/R/weights/best.engine").exists()


def test_export_resumes_from_manifest(tmp_path, monkeypatch):
    fake = install(monkeypatch)
    w = make_weights(tmp_path / "runs" / "R", "R")
    variants = ["onnx_fp32", "trt_fp16"]
    export_model(w, tmp_path / "m", "R", "rgb", variants, imgsz=640)
    n = len(fake.export_calls)
    export_model(w, tmp_path / "m", "R", "rgb", variants, imgsz=640)
    assert len(fake.export_calls) == n
    (tmp_path / "m/R_trt_fp16.engine").unlink()
    export_model(w, tmp_path / "m", "R", "rgb", variants, imgsz=640)
    assert len(fake.export_calls) == n + 1
    assert fake.export_calls[-1]["quantize"] == 16
    assert fake.export_calls[-1]["format"] == "engine"


def test_int8_without_calibration_is_error(tmp_path, monkeypatch):
    install(monkeypatch)
    w = make_weights(tmp_path / "runs" / "R", "R")
    with pytest.raises(ValueError):
        export_model(w, tmp_path / "m", "R", "rgb", ["onnx_int8"], imgsz=640, stems=(), root=None)


def test_trt_int8_without_calib_yaml_is_error(tmp_path, monkeypatch):
    install(monkeypatch)
    w = make_weights(tmp_path / "runs" / "R", "R")
    with pytest.raises(ValueError):
        export_model(w, tmp_path / "m", "R", "rgb", ["trt_int8"], imgsz=640, calib_yaml=None)
