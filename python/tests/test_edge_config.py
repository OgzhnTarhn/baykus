"""Edge config yükleme, varyant kaydı ve model dosya adı testleri (Görev 5)."""

from pathlib import Path

import pytest
import yaml

from baykus.edge.config import load_edge_config
from baykus.edge.variants import VARIANTS, model_file
from baykus.train.config import ConfigError

REPO = Path(__file__).resolve().parents[2]


def test_repo_edge_config():
    cfg = load_edge_config(REPO / "configs/edge/e9_edge.yaml")
    assert cfg.rule.target_p95_ms == 100 and cfg.rule.max_map50_drop == pytest.approx(0.010)
    assert cfg.rule.order == ("e4_late_fusion", "e3c_modality_dropout")
    assert set(cfg.variants) == set(VARIANTS) and cfg.calibration.n == 500
    assert cfg.pipelines["e4_late_fusion"].rgb == "e1_rgb"
    for ref in cfg.models.values():  # run kimlikleri gerçekten var
        assert (REPO / "results/runs" / f"{ref.run_id}_val_ir.json").exists()


@pytest.mark.parametrize(
    "patch",
    [
        {"extra": 1},
        {"variants": ["onnx_int4"]},
        {"pipelines": {"p": {"kind": "single", "model": "yok"}}},
    ],
)
def test_bad_config_is_error(tmp_path, patch):
    data = yaml.safe_load((REPO / "configs/edge/e9_edge.yaml").read_text(encoding="utf-8"))
    data.update(patch)  # pipelines patch'i sözlüğü tamamen değiştirir
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_edge_config(p)


def test_model_file_names():
    assert model_file(Path("m"), "R", "trt_int8") == Path("m/R_trt_int8.engine")
    assert model_file(Path("m"), "R", "onnx_int8") == Path("m/R_onnx_int8.onnx")


def test_variant_registry_order_and_fields():
    assert list(VARIANTS) == [
        "onnx_fp32",
        "onnx_fp16",
        "onnx_int8",
        "onnx_int8_headfp32",
        "trt_fp32",
        "trt_fp16",
        "trt_int8",
    ]
    assert VARIANTS["onnx_int8_headfp32"].head_fp32 is True
    assert VARIANTS["trt_int8"].fmt == "engine"
    assert VARIANTS["trt_int8"].device == "cuda"
    assert VARIANTS["onnx_fp16"].precision == "fp16"
