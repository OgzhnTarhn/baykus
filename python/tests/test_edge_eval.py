"""baykus-edge eval: ORT/TensorRT arka uçları ve varyant doğruluk kayıtları (H5, FR-X4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from baykus.edge.backends import OrtBackend
from baykus.edge.evaluate import EdgeError, evaluate_variant
from baykus.train.config import load_config
from tests.synthetic import SIZE, make_synthetic_dataset

RUN = "2026-10-11_e1-rgb_s_49da0cb"


def _cfg(tmp_path: Path) -> Path:
    p = tmp_path / "e1.yaml"
    p.write_text(
        "experiment: e1_rgb\nmode: rgb\nmodel: yolo26n.pt\ndata: data/dv_rgbt/data_rgb.yaml\n"
        "eval:\n  n_boot: 10\n",
        encoding="utf-8",
    )
    return p


class PerfectBackend:
    """GT kutularını letterbox'lı uzayda geri verir; aldığı dtype'ı kaydeder."""

    def __init__(self, root: Path, split: str = "val", dtype=np.float32):
        coco = json.loads((Path(root) / "coco" / f"{split}_ir.json").read_text(encoding="utf-8"))
        by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        self.outputs: list[np.ndarray] = []
        for img in coco["images"]:
            anns = by_image.get(img["id"], [])
            if anns:
                rows = [
                    [
                        a["bbox"][0],
                        a["bbox"][1],
                        a["bbox"][0] + a["bbox"][2],
                        a["bbox"][1] + a["bbox"][3],
                        0.9,
                        a["category_id"] - 1,
                    ]
                    for a in anns
                ]
                self.outputs.append(np.array([rows], dtype=np.float32))
            else:
                self.outputs.append(np.zeros((1, 0, 6), dtype=np.float32))
        self.input_dtype = np.dtype(dtype)
        self.channels = 3
        self.imgsz = SIZE
        self.provider = "cpu"
        self.threads = None
        self.seen_dtypes: list[np.dtype] = []
        self.i = 0

    def infer(self, x: np.ndarray) -> np.ndarray:
        self.seen_dtypes.append(x.dtype)
        assert x.shape == (1, 3, SIZE, SIZE)
        out = self.outputs[self.i]
        self.i += 1
        return out


def _setup(tmp_path, monkeypatch, dtype=np.float32, calls=None):
    root = make_synthetic_dataset(tmp_path / "dv")
    model = tmp_path / "m" / f"{RUN}_onnx_int8.onnx"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model-a")
    monkeypatch.setattr(
        "baykus.edge.evaluate.open_backend",
        lambda p, threads=None: (
            (calls.append(p) if calls is not None else None) or PerfectBackend(root, dtype=dtype)
        ),
    )
    return root, model


def _eval(tmp_path, root, model, **kw):
    return evaluate_variant(
        model_cfg=load_config(_cfg(tmp_path)),
        run_id=RUN,
        config="configs/e1_rgb.yaml",
        variant="onnx_int8",
        model_path=model,
        data_root=root,
        meta=root / "meta",
        split="val",
        project=tmp_path / "proj",
        results=tmp_path / "res",
        **{"n_boot": 0, **kw},
    )


def test_evaluate_variant_record_and_names(tmp_path: Path, monkeypatch):
    root, model = _setup(tmp_path, monkeypatch)
    paths = _eval(tmp_path, root, model)
    assert [p.name for p in paths] == [f"{RUN}_val_onnx_int8_ir.json"]
    assert paths[0].parent == tmp_path / "res" / "edge"
    rec = json.loads(paths[0].read_text(encoding="utf-8"))
    assert rec["format"] == "onnx" and rec["precision"] == "int8"
    assert rec["split"] == "val" and rec["label_set"] == "ir"
    assert rec["metrics"]["map50"] == pytest.approx(1.0)
    assert rec["metrics"]["ci95"]["map50"] == [None, None]
    assert rec["n_boot"] == 0
    assert rec["edge"]["variant"] == "onnx_int8"
    assert len(rec["edge"]["sha256"]) == 64
    assert rec["edge"]["provider"] == "cpu"
    assert rec["edge"]["file_size_mb"] >= 0
    assert rec["edge"]["seconds_per_image"] >= 0
    assert rec["model"]["mode"] == "rgb"
    assert rec["weights"] == str(model)
    assert (tmp_path / "proj" / f"{RUN}_val_onnx_int8" / "predictions_coco.json.gz").exists()


def test_label_sets_both(tmp_path: Path, monkeypatch):
    root, model = _setup(tmp_path, monkeypatch)
    paths = _eval(tmp_path, root, model, label_sets=("ir", "rgb"))
    assert [p.name for p in paths] == [
        f"{RUN}_val_onnx_int8_ir.json",
        f"{RUN}_val_onnx_int8_rgb.json",
    ]


def test_rerun_skips_and_other_file_is_error(tmp_path: Path, monkeypatch):
    calls: list[Path] = []
    root, model = _setup(tmp_path, monkeypatch, calls=calls)
    first = _eval(tmp_path, root, model)
    second = _eval(tmp_path, root, model)
    assert len(calls) == 1  # aynı sha256 + n_boot → arka uç yeniden açılmaz
    assert second == first
    model.write_bytes(b"model-b")
    with pytest.raises(EdgeError, match="farklı"):
        _eval(tmp_path, root, model)
    model.write_bytes(b"model-a")
    with pytest.raises(EdgeError):
        _eval(tmp_path, root, model, n_boot=5)
    assert issubclass(EdgeError, ValueError)


def test_input_cast_to_backend_dtype(tmp_path: Path, monkeypatch):
    root = make_synthetic_dataset(tmp_path / "dv")
    model = tmp_path / "m" / f"{RUN}_onnx_int8.onnx"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model-a")
    backends: list[PerfectBackend] = []

    def fake_open(p, threads=None):
        b = PerfectBackend(root, dtype=np.float16)
        backends.append(b)
        return b

    monkeypatch.setattr("baykus.edge.evaluate.open_backend", fake_open)
    _eval(tmp_path, root, model)
    assert set(backends[0].seen_dtypes) == {np.dtype(np.float16)}


def test_ort_backend_reads_dtype_channels_threads(tmp_path: Path):
    inp = onnx.helper.make_tensor_value_info("images", TensorProto.FLOAT16, [1, 4, 32, 32])
    out = onnx.helper.make_tensor_value_info("output0", TensorProto.FLOAT16, [1, 4, 32, 32])
    node = onnx.helper.make_node("Identity", ["images"], ["output0"])
    graph = onnx.helper.make_graph([node], "g", [inp], [out])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    p = tmp_path / "f16.onnx"
    onnx.save(model, p)
    b = OrtBackend(p, threads=2)
    assert b.input_dtype == np.float16
    assert b.channels == 4
    assert b.imgsz == 32
    assert b.threads == 2
    assert b.provider == "cpu"
    assert b.infer(np.zeros((1, 4, 32, 32), np.float16)).shape == (1, 4, 32, 32)


def test_cli_eval(tmp_path: Path, monkeypatch):
    from baykus.edge.cli import main

    root, model = _setup(tmp_path, monkeypatch)
    repo = Path(__file__).resolve().parents[2]
    rc = main(
        [
            "eval",
            "--config",
            str(repo / "configs/edge/e9_edge.yaml"),
            "--model",
            "e1_rgb",
            "--variant",
            "onnx_int8",
            "--file",
            str(model),
            "--data-root",
            str(root),
            "--meta",
            str(root / "meta"),
            "--project",
            str(tmp_path / "proj"),
            "--results",
            str(tmp_path / "res"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "res" / "edge" / f"{RUN}_val_onnx_int8_ir.json").exists()
